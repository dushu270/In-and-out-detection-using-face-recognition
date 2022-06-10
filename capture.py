from tkinter.ttk import LabeledScale
import cv2
import numpy as np
import cv2
import os


face_cascade=cv2.CascadeClassifier('test1/cascades/data/haarcascade_frontalface_alt2.xml')
cap=cv2.VideoCapture(0)
def make_1080p():
    cap.set(3, 1920)
    cap.set(4, 1080)

def make_720p():
    cap.set(3, 1280)
    cap.set(4, 720)

def make_480p():
    cap.set(3, 640)
    cap.set(4, 480)

def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)

make_1080p()



def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

i=1
while(True):
    
    ret,frame=cap.read()
    frame=rescale_frame(frame,percent=80)
    frame=cv2.flip(frame,1)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(image=gray,scaleFactor=1.2,minNeighbors=6)   


    for(x1,y1,w1,h1) in faces: 

        roi_gray=gray[y1-60:y1+h1+60 , x1-60:x1+w1+60]
        print(roi_gray)
        img_name="myimage{}.jpg".format(i)
        i+=1
        targetDir="C:/Users/dushu/Desktop/majorproject/test1/images/dushyanth/"
        cv2.imwrite(os.path.join(targetDir,img_name), roi_gray)
    
    
    cv2.imshow('frame',frame)

    if(i==100 or cv2.waitKey(20) & 0xFF ==  ord('q')):
        break
    
        

    

        
cap.release()
cv2.destroyAllWindows()        

        