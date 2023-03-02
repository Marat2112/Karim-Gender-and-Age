import os
import sys
from pathlib import Path
import argparse

import cv2




# Мы программируем функцию highlightFace() для определений координат лица человека.
def highlightFace(net, frame, conf_threshold=0.7):

    # В первых трех строках кода этой функции мы получим пустую копию кадра (frame), с помощью которой мы определим высоту и ширину кадра.
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]

    # Далее мы сконструируем blob (Binary Large Object – большой двоичный объект) и пропустим его через нейронную сеть для обнаружения лица.
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    # После этого в цикле по обнаруженным лицам мы извлечем координаты этих лиц, которые мы будем затем использовать для рисования ограничивающих прямоугольников вокруг лиц.
    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes


# Создаем поведение работы программы с помощью введения
# дополнительного аргумента для парсера, который определяет
# считываются ли изображения с папки или включается видеокамера


def parse_arguments():
    """
    CMD parser settings + parse arguments
    """
    parser=argparse.ArgumentParser()

    parser.add_argument('--work_format', default="imgs", choices=["imgs", "web_camera"], type=str)
    parser.add_argument('--path_to_imgs_folder', default=None, type=str)

    return parser.parse_args()


# Ниже стандартные конфиги типа названий и констант по смыслу


###
# Configs

# faceNet pretrained weights path
faceProto = "config/opencv_face_detector.pbtxt"
faceModel = "config/opencv_face_detector_uint8.pb"
# faceNet pretrained weights path
ageProto = "config/age_deploy.prototxt"
ageModel = "config/age_net.caffemodel"
# faceNet pretrained weights path
genderProto = "config/gender_deploy.prototxt"
genderModel = "config/gender_net.caffemodel"

# Конфигурационная модель
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = [
    '(0-2)',
    '(4-6)', 
    '(8-12)', 
    '(15-20)', 
    '(25-32)', 
    '(38-43)', 
    '(48-53)', 
    '(60-100)'
]
genderList = [
    'Male', 
    'Female'
]
###



# Чтобы какой-то код исполнялся в программе, то всегда стоит поступать через проверку
# имени исполняемого файла __name__ == "__main__"

if __name__ == "__main__":
    # Parse cmd arguments 
    args = parse_arguments()


    # Инициализация модели
    faceNet=cv2.dnn.readNet(faceModel,faceProto)
    ageNet=cv2.dnn.readNet(ageModel,ageProto)
    genderNet=cv2.dnn.readNet(genderModel,genderProto)


    # Здесь мы разветвляем логику работы программы,
    # чтобы в зависимости от параметра, переданного через
    # командную строку, программа работала по разному

    # Формат работы: подаётся путь до папки с изображениями
    if args.work_format == "imgs":
    # Проверка того, что путь до папки был передан
        assert args.path_to_imgs_folder is not None, "args: path_to_imgs_folder should be not empty, please write path to folder with images"
        # Формируем список файлов внутри папки
        abs_path_to_img_folder = os.path.abspath(args.path_to_imgs_folder)
        imgs_in_folder = os.listdir(abs_path_to_img_folder)
        # Сформировали список абсолютных путей до каждого изображения в папке
        list_with_abs_path_to_images = [str(Path(abs_path_to_img_folder, img_name)) for img_name in imgs_in_folder]
        # Итерируемся по каждому пути в списке всех путей до изображений, обрабатываем
         # каждое друг за другом и сохраняем каждое обработанное изображение в папке /results к примеру
        for i, abs_path_to_img in enumerate(list_with_abs_path_to_images):
            img_i = cv2.imread(abs_path_to_img)
            # читаем изображение с помощью библиотеки OpenCV 
            # обрабатываем избражение с помощью моделей, результат сохраняем
            padding = 20
            ImageRes, faceBoxes = highlightFace(faceNet, img_i)

            if not faceBoxes:    
                print(f"{abs_path_to_img}: No face detected")
            else:
                for faceBox in faceBoxes:
                    face = img_i[max(0, faceBox[1] - padding):
                                    min(faceBox[3] + padding, img_i.shape[0] - 1), max(0, faceBox[0] - padding)
                                    :min(faceBox[2] + padding, img_i.shape[1] - 1)]
                    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                    genderNet.setInput(blob)
                    genderPreds = genderNet.forward()
                    gender = genderList[genderPreds[0].argmax()]
                    print(f'Gender: {gender}')

                    ageNet.setInput(blob)
                    agePreds = ageNet.forward()
                    age = ageList[agePreds[0].argmax()]
                    print(f'Age: {age[1:-1]} years')

                    cv2.putText(ImageRes, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 255, 255), 2, cv2.LINE_AA)

                    cv2.imwrite(f"results/img_{i}.jpg", ImageRes)

                    print(f"{abs_path_to_img}: save result in img_{i}.jpg")

    # Формат работы: веб камера
    elif args.work_format == "web_camera":
        video = cv2.VideoCapture(0)
        padding = 20
        while cv2.waitKey(1) < 0:
            hasFrame, frame = video.read()
            if not hasFrame:
                cv2.waitKey(0) == ord('q')
                sys.exit(0)

            ImageRes, faceBoxes = highlightFace(faceNet, frame)
            if not faceBoxes:
                    print("No face detected")

            for faceBox in faceBoxes:
                face = frame[max(0, faceBox[1] - padding):
                                min(faceBox[3] + padding, frame.shape[0] - 1), max(0, faceBox[0] - padding)
                                :min(faceBox[2] + padding, frame.shape[1] - 1)]
                blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                genderNet.setInput(blob)
                genderPreds = genderNet.forward()
                gender = genderList[genderPreds[0].argmax()]
                print(f'Gender: {gender}')

                ageNet.setInput(blob)
                agePreds = ageNet.forward()
                age = ageList[agePreds[0].argmax()]
                print(f'Age: {age[1:-1]} years')

                cv2.putText(ImageRes, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 255, 255), 2, cv2.LINE_AA)

                cv2.imwrite("results/imageWebCam.jpg", ImageRes)
                print("save result in imageWebCam.jpg")

                


                

        















