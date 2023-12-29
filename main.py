import cv2
import os
from detection.opencv import *
from detection.noopencv import *
from detection.geometry import *
from tracking.tracking import ObjectTracking

# Имя выходного видеофайла
output_name = "output.avi"

# Получаем путь к директории, в которой находится скрипт
dir_path = os.path.dirname(os.path.abspath(__file__))

# Открываем видеофайл, находящийся в той же директории, что и скрипт
video = cv2.VideoCapture(os.path.join(dir_path, "people.avi"))
#video = cv2.VideoCapture(os.path.join(dir_path, "cars.mp4"))

# Проверяем, удалось ли открыть видеофайл
if video.isOpened() == False:
    print('Не возможно открыть файл')

# Считываем первый кадр видеофайла
ret,frame = video.read()

# Создаем объекты для обнаружения и отслеживания объектов
detection = OpenCVDetection()
#detection = NoOpenCVDetection()
tracking = ObjectTracking()

# Проверяем, удалось ли считать первый кадр видеофайла
if  not ret:
    raise RuntimeError("Cannot acces video stream")

# Цикл обработки кадров видеофайла
while video.isOpened():
    # Проверяем, удалось ли считать кадр видеофайла
    if frame is None:
        break

    # Изменяем размер кадра до заданных размеров
    frame = cv2.resize(frame, (768 , 576))

    # Обнаруживаем объекты на кадре
    frame_det = detection.update(frame)

    # Отслеживаем объекты на кадре
    frame_tr = tracking.run(frame)

    # Отображаем кадр с результатами обнаружения и отслеживания объектов
    cv2.imshow("Cam", frame_tr)

    # Считываем следующий кадр видеофайла
    ret, frame = video.read()

    # Обработка нажатий клавиш
    button = cv2.waitKey(25)
    if button == 27:
        break

# Освобождаем ресурсы, связанные с видеофайлом
video.release()

# Закрываем все окна OpenCV
cv2.destroyAllWindows()
