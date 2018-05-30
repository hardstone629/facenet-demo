import cv2
import sys
import os
import detect_face
import facenet
import face as faceLib
import tensorflow as tf

def recognition(source):
    frame_interval = 1 
    frame_count = 0
    video_capture = cv2.VideoCapture(source)
    face_detect = faceLib.Detection()
    face_encoder = faceLib.Encoder()
    # workpath = os.getcwd()+"/dataset/orignal/"+perName
    # if os.path.exists(workpath)!=True:
    #     os.makedirs( workpath )
    # num=1
    face_identifer = faceLib.Identifier()
    while True:
        ret, frame = video_capture.read()
        orignalFrame = frame
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        if (frame_count % frame_interval)== 0:
            faces = face_detect.find_faces(frame)
            for face in faces:
                # if num > maxNum:
                #     break;
                
                face_bb = face.bounding_box.astype(int)
                left = face_bb[0]
                top = face_bb[1]
                right = face_bb[2]
                bottom = face_bb[3]
                cv2.rectangle(frame,(left,top), (right, bottom),(0, 255, 0), 2)
                # img_name = '%s/%d.jpg'%(workpath, num)
                # cv2.imwrite(img_name,face.image,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
                # num += 1
                face.embedding = face_encoder.generate_embedding(face)
                clsname = face_identifer.identify(face)
                face.name = clsname
                print('classname="%s"' % clsname)

                cv2.rectangle(frame, (left,bottom-35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, clsname, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            cv2.imshow('video',frame)
        frame_count+=1    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break;
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    recognition(0)
    