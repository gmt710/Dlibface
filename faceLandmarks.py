# encoding:utf-8

import dlib
import numpy as np
import cv2
import os
import time

def rect_to_bb(rect): # 获得人脸矩形的坐标信息
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

def shape_to_np(shape, is68Landmarks=True, dtype="int"): # 将包含68个特征的的shape转换为numpy array格式
    if is68Landmarks:
        landmarkNum = 68
    else:
        landmarkNum = 5
    coords = np.zeros((landmarkNum, 2), dtype=dtype)
    for i in range(0, landmarkNum):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def resize(image, width=1200):  # 将待检测的image进行resize
    r = width * 1.0 / image.shape[1]
    dim = (width, int(image.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

def feature(is68Landmarks=True):
    image_path = "./data/"
    image_file = "test_1988.jpg"
    detector = dlib.get_frontal_face_detector()
    if is68Landmarks:
        predictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")
    else:
        predictor = dlib.shape_predictor("./models/shape_predictor_5_face_landmarks.dat")

    image = cv2.imread(image_path + image_file)
    image = resize(image, width=1200)# 1200
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    shapes = []
    startTime = time.time()
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = shape_to_np(shape, is68Landmarks)
        shapes.append(shape)
        (x, y, w, h) = rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, "Face: {}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    print("{} method, detect spend {}s ".format(("68Landmarks" if is68Landmarks else "5Landmarks"), time.time()-startTime))

    for shape in shapes:
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
    cv2.imshow("Output", image)
    savePath = "./results/landmarks/"
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    if is68Landmarks:
        saveName = image_file[:-4] + "_68Landmarks.jpg"
    else:
        saveName = image_file[:-4] + "_5Landmarks.jpg"
    cv2.imwrite(savePath + saveName, image)
    cv2.waitKey(10)

if __name__ == "__main__":
    is68Landmarks = True
    feature(is68Landmarks)
    if is68Landmarks:
        is68Landmarks = not is68Landmarks
        feature(is68Landmarks)
