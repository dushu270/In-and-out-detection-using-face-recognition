from tkinter.ttk import LabeledScale
import cv2
import numpy as np
import cv2
import pickle
import datetime

face_cascade=cv2.CascadeClassifier('test1/cascades/data/haarcascade_frontalface_alt2.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels={}
with open("labels.pickle","rb") as f:
    og_labels=pickle.load(f)
    labels={v:k for k,v in og_labels.items()}

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
#change_res(1280, 720)


def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)




while(True):
    
    ret,frame=cap.read()
    frame=rescale_frame(frame,percent=80)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(image=gray,scaleFactor=1.5,minNeighbors=5)

    for(x,y,w,h) in faces:
        #print(x,y,w,h)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]
        
        id_,conf=recognizer.predict(roi_gray)

        if(conf>=50):
            print(conf)
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name=labels[id_]
            color=(255,255,255)
            stroke=2
            cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)

        img_item="myimage.jpg"
        cv2.imwrite(img_item,roi_gray)
        color=(255,0,0)
        stroke=2
        width=x+w
        height=y+h
        cv2.rectangle(frame, (x,y),(width,height),color,stroke) 
    cv2.imshow('frame',frame)   
    if(cv2.waitKey(20) & 0xFF ==  ord('q')):
        break    
    
cap.release()
cv2.destroyAllWindows()    
    
    
   