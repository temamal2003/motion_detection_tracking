import numpy as np
import cv2

class ObjectTracking:
    def __init__(self):
        self.track_len = 10 # Длина трека
        self.detect_interval = 5 # Интервал, на котором происходит поиск новых особенностей
        self.tracks = [] # Список треков
        self.frame_idx = 0 # Индекс кадра
        # Параметры для оптического потока Лукаса-Канаде
        self.lk_params = dict( winSize  = (15, 15), 
                  maxLevel = 2, 
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))    
        # Параметры для поиска особенностей
        self.feature_params = dict( maxCorners = 500, 
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

    def run(self,frame):
        while True:
            # Преобразование кадра в оттенки серого
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Создание копии кадра
            vis = frame.copy()

            # Если есть существующие треки, используйте оптический поток Лукаса-Канаде для их отслеживания
            if len(self.tracks) > 0:
                # Получение предыдущего и текущего кадров
                img0, img1 = self.prev_gray, frame_gray
                # Получение предыдущих точек
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                # Расчет новых точек с использованием оптического потока Лукаса-Канаде
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.lk_params)
                # Расчет обратного потока
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.lk_params)
                # Расчет разницы между исходным и обратным потоком
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                # Определение, какие точки хорошие
                good = d < 1
                # Создание списка новых треков
                new_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    # Рисование круга в новой точке
                    cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), cv2.FILLED)
                # Обновление списка треков
                self.tracks = new_tracks
                # Рисование треков
                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0),thickness=2)

            # Если пришло время обнаружить новые особенности, то делаем это
            if self.frame_idx % self.detect_interval == 0:
                # Создание маски для кадра
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                # Добавление кругов на маску в конце каждого трека
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, cv2.FILLED)
                # Обнаружение новых особенностей с использованием маски
                p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **self.feature_params)
                # Если обнаружены новые особенности, добавляем их в список треков
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])

            # Обновление индекса кадра и предыдущего кадра
            self.frame_idx += 1
            self.prev_gray = frame_gray
            # Возвращение измененного кадра
            return vis