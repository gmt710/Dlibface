# encoding:utf-8

import dlib
import cv2
import numpy as np
import os, glob


def resize(image, width=1200):  # 将待检测的image进行resize
    r = width * 1.0 / image.shape[1]
    dim = (width, int(image.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

def rect_to_bb(rect): # 获得人脸矩形的坐标信息
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

def create_face_space():

    # 对文件夹下的每一个人脸进行:
    # 1.人脸检测
    # 2.关键点检测
    # 3.描述子提取

    # 候选人脸文件夹
    faces_folder_path = "./data/candidate-faces/"
    # 候选人脸描述子list
    descriptors = []
    candidates = []
    for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
        print("Processing file: {}".format(f))
        img = cv2.imread(f)
        # img = resize(img, width=300)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 1.人脸检测
        dets = detector(img, 1)
        print("Number of faces detected: {}".format(len(dets)))
        candidate = f.split('\\')[-1][:-4]
        for k, d in enumerate(dets):
            # 2.关键点检测
            shape = sp(img, d)

            # 3.描述子提取，128D向量
            face_descriptor = facerec.compute_face_descriptor(img, shape)

            # 转换为numpy array
            v = np.array(face_descriptor)
            descriptors.append(v)
            candidates.append(candidate)
    return descriptors, candidates


def predict(descriptors, path):
    # 对需识别人脸进行同样处理
    # 提取描述子
    img = cv2.imread(path)
    # img = io.imread(path)
    # img = resize(img, width=300)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = detector(gray, 1)
    dist = []
    if len(dets) == 0:
        pass
    for k, d in enumerate(dets):
        shape = sp(img, d)
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        d_test = np.array(face_descriptor)

        # 计算欧式距离
        for i in descriptors:
            dist_ = np.linalg.norm(i-d_test)
            dist.append(dist_)
            # print(dist)
    return dist

def demo():
    global detector, sp, facerec
    # 加载正脸检测器
    detector = dlib.get_frontal_face_detector()

    # 加载人脸关键点检测器
    sp = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")

    # 3. 加载人脸识别模型
    facerec = dlib.face_recognition_model_v1("./models/dlib_face_recognition_resnet_model_v1.dat")

    # 提取候选人特征与候选人名单
    descriptors, candidates = create_face_space()
    savePath = "./results/recongnition/"
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    fp = open(savePath + 'recognition_reslut.txt', 'a')
    predict_path = "./data/faces/*.jpg"
    for f in glob.glob(predict_path):
        f = f.replace("\\", '/')
        # print("f :{}".format(f))
        dist = predict(descriptors, f)
        # 候选人和距离组成一个dict
        c_d = dict(zip(candidates, dist))
        if not c_d:
            print(str(c_d) + " is None")
            continue
        cd_sorted = sorted(c_d.items(), key=lambda d:d[1])
        print("c_d :{}".format(cd_sorted))

        print("The person_test--{} is: ".format(f), cd_sorted[0][0])
        fp.write("\nThe person_test--{} is: with similar : {}".format(f, cd_sorted[0][0]))
    fp.close()

if __name__ == "__main__":

    demo()

