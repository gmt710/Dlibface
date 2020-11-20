# encoding:utf-8

import dlib
import cv2
import os
import time

def rect_to_bb(rect): # 获得人脸矩形的坐标信息
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

def resize(image, width=1200):  # 将待检测的image进行resize
    r = width * 1.0 / image.shape[1]
    dim = (width, int(image.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

def detect(isHOG=False):
    image_path = "./data/"
    image_file = "test_1988.jpg"
    startTime = time.time()
    if isHOG:
        detector = dlib.get_frontal_face_detector()  # 基于HOG+SVM分类
    else:
        model_path = "./models/mmod_human_face_detector.dat"  # 基于 Maximum-Margin Object Detector 的深度学习人脸检测方案
        detector = dlib.cnn_face_detection_model_v1(model_path)
    image = cv2.imread(image_path + image_file)
    image = resize(image, width=1200)
    # image = resize(image, width=600)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    print("{} method, detect spend {}s ".format(("HOG" if isHOG else "MMOD"), time.time()-startTime))
    for (i, rect) in enumerate(rects):
        if isHOG:
            (x, y, w, h) = rect_to_bb(rect)
        else:
            (x, y, w, h) = rect_to_bb(rect.rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, "Face： {}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Output", image)
    savePath = "./results/detect/"
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    if isHOG:
        saveName = image_file[:-4] + "_HOG.jpg"
    else:
        saveName = image_file[:-4] + "_MMOD.jpg"
    cv2.imwrite(savePath + saveName, image)
    cv2.waitKey(10)

if __name__ == "__main__":
    isHOG = True
    detect(isHOG)
    if isHOG:
        isHOG = not isHOG
        detect(isHOG)
