import cv2
import numpy as np



# Функция для проверки минимального расстояния между двумя контурами
def check_dist_contours(first, second):
    min_distance = float('inf')
    for p1 in first:
        for p2 in second:
            distance = np.linalg.norm(p1 - p2)
            min_distance = min(distance, min_distance)
            
    return min_distance

# Функция для проверки того, находится ли один прямоугольник внутри другого
def is_inner_rectangle(first, second):
    x1, y1, w1, h1 = first
    x2, y2, w2, h2 = second
    return (x2 >= x1 and y2 >= y1 and x2 + w2 <= x1 + w1 and y2 + h2 <= y1 + h1)

# Функция для определения расстояния между центрами двух прямоугольников
def find_distance_rectangles(first, second):
    center1 = (first[0] + first[2] // 2, first[1] + first[3] // 2)
    center2 = (second[0] + second[2] // 2, second[1] + second[3] // 2)
    return np.linalg.norm(np.array(center1) - np.array(center2))

# Функция для объединения двух прямоугольников
def merge_rectangles(first, second):
    x = min(first[0], second[0])
    y = min(first[1], second[1])
    w = max(first[0] + first[2], second[0] + second[2]) - x
    h = max(first[1] + first[3], second[1] + second[3]) - y
    return (x, y, w, h)

# Функция для проверки того, находится ли точка внутри прямоугольника
def is_inner_point(point, rect):
    
    x, y = point
    x1,y1, x2,y2 = rect

    return x1 <= x <= x2 and y1 <= y <= y2