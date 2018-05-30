import cv2
import os
import sys

def record(fileName):
    videoCapture = cv2.VideoCapture(0)
    (filepath,tempfilename) = os.path.split(fileName)
    if os.path.exists(filepath)!=True:
        os.makedirs(filepath)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)  
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))  
    fourcc = int(videoCapture.get(cv2.CAP_PROP_FOURCC ))
    out = cv2.VideoWriter(fileName, fourcc, fps, size)
    while True:
        ret,frame = videoCapture.read()
        cv2.imshow("frame", frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    videoCapture.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    filename = input("please input recordFileName:")
    record(filename)

     