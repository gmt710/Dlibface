# encoding:utf-8

import dlib
import cv2
import numpy as np
import math
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

def face_alignment_68(faces):
    # 使用68点关键点模型，根据关键点信息求解变换矩阵，然后把变换矩阵应用到整个图像上。
    predictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat") # 用来预测关键点
    faces_aligned = []
    global startTime
    startTime = time.time()
    for face in faces:
        rec = dlib.rectangle(0,0,face.shape[0],face.shape[1])
        shape = predictor(np.uint8(face),rec) # 注意输入的必须是uint8类型
        order = [36,45,30,48,54] # left eye, right eye, nose, left mouth, right mouth  注意关键点的顺序，这个在网上可以找
        for j in order:
            x = shape.part(j).x
            y = shape.part(j).y
            cv2.circle(face, (x, y), 2, (0, 0, 255), -1)

        eye_center =((shape.part(36).x + shape.part(45).x) * 1./2, # 计算两眼的中心坐标
                      (shape.part(36).y + shape.part(45).y) * 1./2)
        dx = (shape.part(45).x - shape.part(36).x) # note: right - right
        dy = (shape.part(45).y - shape.part(36).y)

        angle = math.atan2(dy,dx) * 180. / math.pi # 计算角度
        RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1) # 计算仿射矩阵
        RotImg = cv2.warpAffine(face, RotateMatrix, (face.shape[0], face.shape[1])) # 进行仿射变换，即旋转
        faces_aligned.append(RotImg)
    return faces_aligned


def face_alignment_5(rgb_img, faces):
    startTime = time.time()
    faces_aligned = []
    for face in faces:
        # RotImg = dlib.get_face_chip(rgb_img, face)
        RotImg = dlib.get_face_chip(np.uint8(rgb_img), np.uint8(face))
        # RotImg = dlib.get_face_chip(rgb_img, face, size=224, padding=0.25)
        faces_aligned.append(RotImg)
    return faces_aligned

def demo(isAlignment_5=True):
    image_path = "./data/"
    image_file = "test_1988.jpg"
    im_raw = cv2.imread(image_path + image_file).astype('uint8')

    # detector = dlib.get_frontal_face_detector()
    model_path = "./models/mmod_human_face_detector.dat"  # 基于 Maximum-Margin Object Detector 的深度学习人脸检测方案
    detector = dlib.cnn_face_detection_model_v1(model_path)
    im_raw = resize(im_raw, width=1200)
    gray = cv2.cvtColor(im_raw, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    src_faces = []
    for (i, rect) in enumerate(rects):
        (x, y, w, h) = rect_to_bb(rect.rect)
        detect_face = im_raw[y:y+h,x:x+w]
        src_faces.append(detect_face)
        cv2.rectangle(im_raw, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(im_raw, "Face: {}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    if isAlignment_5:
        faces_aligned = face_alignment_5(im_raw, src_faces)
    else:
        faces_aligned = face_alignment_68(src_faces)
    print("{} method, detect spend {}s ".format(("Alignment_5" if isAlignment_5 else "Alignment_68"), time.time()-startTime))

    cv2.imshow("src", im_raw)
    savePath = "./results/alignment/"
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    if isAlignment_5:
        saveName = "_Align5.jpg"
    else:
        saveName = "_Align68.jpg"
    i = 0
    for face in faces_aligned:
        cv2.imshow("det_{}".format(i), face)
        cv2.imwrite(savePath + image_file[:-4] + "_{}".format(i) + saveName, face)
        i = i + 1
    cv2.waitKey(10)

if __name__ == "__main__":
    isAlignment_5 = False
    demo(isAlignment_5)
    if isAlignment_5:
        isAlignment_5 = not isAlignment_5
        demo(isAlignment_5)

