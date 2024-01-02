import cv2 #для воспроизведения видео, преобразования оттенков серого в цвет, рисования прямоугольников
import numpy as np #для матричных операций
import math
from detection.geometry import *



class NoOpenCVDetection():
    def __init__(self, detection_step=(10, 10), object_selection_step=(3, 3), move_threshold=30, object_threshold=5):
        self.frame = None
        self.detection_step = detection_step
        self.object_selection_step = object_selection_step
        self.move_threshold = move_threshold
        self.object_threshold = object_threshold
       
    

    def compare_images(self, frame1, frame2):
        # функция сравнения двух кадров
        # возвращает кадр с красной маркировкой на местах, где были изменения
   


        if len(frame1) == len(frame2):

            total_pixels = len(frame1)
            pixel_rows = frame1.shape[0]
            pixel_columns = frame1.shape[1]
            image_increment = int(round(float(total_pixels) / 90))

        else:
            return False
        
        frame_change = frame2.copy()

    #Циклы проверки пикселей
        for i in range(1,pixel_rows-1,3):
            for j in range(1,pixel_columns-1,3):
                #Вычисление среднего состояния путем взятия средних значений размером 3x3 пикселя в каждом кадре
                mean_condition= ((abs(frame2.item(i - 1, j,0) - frame1.item(i - 1, j,0)) + abs(
                frame2.item(i, j + 1,0) - frame1.item(i, j + 1,0)) +
                abs(frame2.item(i , j-1,0) - frame1.item(i, j-1,0)) + abs(frame2.item(i + 1, j,0) - frame1.item(i + 1, j,0)) + (abs(frame2.item(i , j,0) - frame1.item(i, j,0))
                )))/5

                #проверка, превышает ли среднее условие в этом местоположении пороговое значение
                if (mean_condition > 5+image_increment):
                    frame_change.itemset((i+1, j + 1,2), 255)

                    frame_change.itemset((i, j - 1,2), 255)
                    frame_change.itemset((i - 1, j,2), 255)
                    frame_change.itemset((i, j + 1,2), 255)
                    frame_change.itemset((i + 1, j,2), 255)
                    frame_change.itemset((i, j,2), 255)
                    frame_change.itemset((i-1,j-1,2),255)
                    frame_change.itemset((i - 1, j + 1,2), 255)
                    frame_change.itemset((i + 1, j - 1,2), 255)

        return frame_change #рамка с красной маркировкой
    
    
    #Проверка сторон и приведенных выше значений для создания прямоугольника наилучшего размера
    def CheckDiagonals(self,frame_changes,x,y,total_rows,total_columns):
        flag = True #используется, чтобы избежать повторений
        xmax = 0
        ymax = 0
        
        theStack = [(y, x)]
        while len(theStack) > 0 and (x<total_columns) and (y<total_rows): # циклическая проверка соседей по координатам стека
            y, x = theStack.pop() #получение координат из стека

            if (((frame_changes.item(y+1,x,2)) !=255 and (frame_changes.item(y,x+1,2)) != 255)): #если true, то конец кластера

                return xmax,ymax #возвращаем ширину и высоту

            if frame_changes.item(y+1, x, 2) == 255 and (flag):
                flag = False #не допускает повторной проверки
                ymax+=1 #увеличить высоту
                theStack.append((y+1,x)) #добавляет следующую координату в стек для проверки

            else:
                flag = True
                xmax+=1 #увеличьте ширину
                theStack.append((y,x+1)) #добавьте следующую координату в стек для проверки


    # функция для объединения перекрывающихся прямоугольников
    def cluster_find(self,boxes, overlapThresh):
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")
        
        pick = []

        #координаты ограничивающих прямоугольников
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]

        # вычислите площадь ограничивающих прямоугольников и отсортируйте ограничивающие
        #  прямоугольники по нижней правой y-координате ограничивающего прямоугольника
        area = (x2 - x1 + 1) * (y2 - y1 + 1)

        idxs = np.argsort(y2) 

        #print(idxs)

        while len(idxs) > 0:
            
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # найдите наибольшие (x, y) координаты для начала
            # ограничивающего прямоугольника и наименьшие (x, y) координаты
            # для конца ограничивающего прямоугольника, перебирая индексы
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # ширина и высота ограничивающего прямоугольника
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # Коэффициент перекрытия
            overlap = (w * h) / area[idxs[:last]]

            # удалите все индексы из списка индексов, которые не подпадают под порог перекрытия
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
            # возвращает только те ограничивающие рамки, которые были выбраны

        return boxes[pick].astype("int")
    
    # функция для нахождения центра прямоугольника
    def findCenter(self, rect ):

        return rect[0] + (rect[2] - rect[0] )// 2, rect[1] + (rect[3]-rect[1])//2
    
    # функция для нахождения расстояния между центрами двух прямоугольников
    def findDistanceRectangles(self, first, second):
        x1, y1 = self.findCenter(first)
        x2, y2 = self.findCenter(second)
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # функция для объединения двух прямоугольников в один
    def mergeRectangles(self,first, second):
        first[0] = min(first[0], second[0])
        first[1] = min(first[1], second[1])
        first[2] = max(first[2], second[2]) 
        first[3] = max(first[3], second[3])

    # функция для объединения перекрывающихся прямоугольников в списке
    def mergeRectanglesList(self,rectangles,step):
        def isShortDistance(first, second):
            min_dist = min((first[2] - first[0] )// 2 + (second[2] - second[0] ) // 2, (first[3]-first[1])// 2 + (second[3]-second[1]) // 2)
            return self.findDistanceRectangles(first, second) < min_dist + 3 * step

        for i in range(len(rectangles) - 1):
            for j in range(i + 1, len(rectangles)):
                if isShortDistance(rectangles[i], rectangles[j]):
                    self.mergeRectangles(rectangles[i], rectangles[j])
        return

    # функция для обновления кадра и поиска объектов на нем
    def update(self, frame):
        frame_copy=frame.copy()
        rectangle_list = []
        if self.frame is not None:
            frame_changes = (self.compare_images(self.frame,frame))
            total_pixels = len(frame_changes)
            pixel_rows = frame_changes.shape[0]
            pixel_columns = frame_changes.shape[1]
            image_increment = int(round(float(total_pixels) / 90)) 

            
            for i in range(3,pixel_columns-7,5):
                for j in range(3,pixel_rows-7,5):
                    
                    if frame_changes.item(j,i,2) ==255:
                        #вызов метода поиска кластера
                        max = self.CheckDiagonals(frame_changes,i,j, pixel_rows-10, pixel_columns-10)                  
                        rec =0
                        if max == None:
                            max = [0,0]
                        xmax = max[1]
                        ymax = max[0]
                        #cv2.rectangle(frame, (i, j), (i+max[0], j+max[1]), (255, 255, 0), 2)
                        if xmax + ymax > image_increment**2:
                            rectangle = i, j, i + ymax, j + xmax
                            #создание прямоугольного списка для поиска больших кластеров
                            rectangle_list.append(rectangle)           
            rectangle_list = np.array(rectangle_list)
            rectangle_list.astype('float')
            #создание плавающих элементов для правильного вычисления перекрытия
            #вызов cluster_find для удаления перекрывающихся / внутренних прямоугольников
            self.mergeRectanglesList(rectangle_list,15)
            rectList = self.cluster_find(rectangle_list,0)
            for p in  rectList: #rectlist содержит 4 координаты для каждого прямоугольника
                cv2.rectangle(frame, (p[0], p[1]), (p[2], p[3]), (255, 255, 0), 2)
        self.frame=frame_copy
