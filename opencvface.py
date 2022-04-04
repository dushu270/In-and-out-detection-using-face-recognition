from tkinter.ttk import LabeledScale
import cv2
import numpy as np
import cv2
import pickle
import datetime
import time


left, center, right = False, False, False
x=150
mask = np.zeros((1000,1000,3))
face_cascade=cv2.CascadeClassifier('test1/cascades/data/haarcascade_frontalface_alt2.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("test1/trainer.yml")
ytest_pos = 40

labels={}
with open("test1/labels.pickle","rb") as f:
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
    frame=cv2.flip(frame,1)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(image=gray,scaleFactor=1.5,minNeighbors=5)

    for(x1,y1,w1,h1) in faces:

        #print(x,y,w,h)
        x=x1
        roi_gray=gray[y1:y1+h1,x1:x1+w1]
        roi_color=frame[y1:y1+h1,x1:x1+w1]
        
        id_,conf=recognizer.predict(roi_gray)

        if(conf<=100):
            print(conf)
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name=labels[id_]
            color=(255,255,255)
            stroke=2
            cv2.putText(frame,name,(x1,y1),font,1,color,stroke,cv2.LINE_AA)



        if not(left) and not(right):
            if x < 100:
                left = True

            elif x > 300:
                right = True    

                
        elif left : 
            if x < 300 and x > 150 and not(center):
                print(x," x has been set to it")
                center = True

            elif x > 300:
                if center:
                    print("motion to right taken place")
                    '''outp = np.zeros((1920, 500,300))
                    cv2.putText(outp, ("gone at {}".format(datetime.datetime.now().strftime("%H:%M"))+name),(0,0),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255))'''
                    #mask = np.zeros((1000, 1000))
                    ytest_pos+=50
                    cv2.putText(mask, ("gone at {}".format(datetime.datetime.now().strftime("%H:%M"))+name), (50, ytest_pos), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)
                    left = False
                    center = False
                else:
                    right = True

        elif right : 
            if x < 400 and x > 150 and not(center):
                print(x," x has been set to it")
                center = True
            if x < 100:
                if center:
                    print("motion done to the left side")
                    ''' outp = np.zeros((1920,500, 300))
                    cv2.putText(outp, ("came at {}".format(datetime.datetime.now().strftime("%H:%M"))+name),(0,0),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255))'''

                    #mask = np.zeros((1920, 500,300))
                    #mask = np.zeros((1000, 1000))
                    ytest_pos+=50
                    cv2.putText(mask, ("came at {}".format(datetime.datetime.now().strftime("%H:%M"))+name), (50, ytest_pos), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
                    right = False
                    center = False

                else:
                    left = True        
            
            

        #img_item="test1/myimage.jpg"
        


        #cv2.imwrite(img_item,roi_gray)
        color=(255,0,0)
        stroke=2
        width=x1+w1
        height=y1+h1
        cv2.rectangle(frame, (x1,y1),(width,height),color,stroke) 
    cv2.imshow('frame',frame)
    cv2.imshow("mask", mask)   
    if(cv2.waitKey(20) & 0xFF ==  ord('q')):
        break    
    
cap.release()
cv2.destroyAllWindows()    
    
    
   