import cv2
import facenet
import classifier
import detect_face
import face
import sys
import os
import catchFromCamera
import recognitionFromCamera as rg


if __name__ == '__main__':
    name = input("请输入需要训练的人员姓名:")
    trainVedio = input("请输入仅包含训练人员的视频文件全名（包含路径）:")
    testVedio = input("请输入需要测试的视频文件全名（包含路径）:")
    if os.path.exists(trainVedio)==False:
        print("训练视频文件不存在！")
        exit(-1)
    
    if os.path.exists(trainVedio)==False:
        print("测试视频文件不存在！")
        exit(-1)

    catchFromCamera.catch(name,100,trainVedio)
    cls = classifier.Classifier()
    cls.train()
    rg.recognition(testVedio)
    # catch(name,face_num_max,"camera")