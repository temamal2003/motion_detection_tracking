import cv2
import numpy as np

class OpenCVDetection():
    def __init__(self):
        # Создание объекта вычитания фона для выделения переднего плана из фона.
        # Включение обнаружения теней.
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

        # Создание ядер для эрозии и дилатации.
        self.erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def update(self, frame):
        # Применение вычитания фона, чтобы получить маску переднего плана.
        fg_mask = self.bg_subtractor.apply(frame)

        # Пороговая обработка маски: все значения пикселей выше 244 будут установлены в 255 (белый).
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Применение эрозии для удаления мелких деталей из маски.
        cv2.erode(thresh, self.erode_kernel, thresh, iterations=2)

        # Применение дилатации для восстановления формы объекта после эрозии.
        cv2.dilate(thresh, self.dilate_kernel, thresh, iterations=2)

        # Нахождение контуров на пороговом изображении. Использование только внешних контуров.
        # CHAIN_APPROX_SIMPLE сокращает количество точек в контуре.
        contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Проход по каждому контуру.
        for c in contours:
            # Отбрасывание контуров малого размера.
            if cv2.contourArea(c) > 1000:
                # Расчет ограничивающего прямоугольника для контура.
                x, y, w, h = cv2.boundingRect(c)
                # Рисование прямоугольника вокруг объекта на исходном кадре.
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

        # Возвращение кадра с нарисованными прямоугольниками вокруг обнаруженных объектов.
        return frame