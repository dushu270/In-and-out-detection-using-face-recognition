import os
from PIL import Image
import numpy as np
import cv2
import pickle
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
image_dir= os.path.join(BASE_DIR,"testimages")
face_cascade=cv2.CascadeClassifier('test1/cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade=cv2.CascadeClassifier('test1/cascades/data/haarcascade_eye.xml')

recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("test1/trainer.yml")
labels={}
with open("test1/labels.pickle","rb") as f:
    og_labels=pickle.load(f)
    labels={v:k for k,v in og_labels.items()}
print(labels)    


for root,dirs,files in os.walk(image_dir):
    for file in files:
        if(file.endswith("jpg") or file.endswith("png") or file.endswith("jpeg")):
            path=os.path.join(root,file)
            label=os.path.basename(os.path.dirname(path)).replace(" ","").lower()
            
            pil_image=Image.open(path).convert("L")
            size=(300,300)
            final_image=pil_image.resize(size,Image.ANTIALIAS)
            #print(final_image)
            image_array=np.array(pil_image,"uint8")
            #print(image_array)
            faces=face_cascade.detectMultiScale(image=image_array,scaleFactor=1.2,minNeighbors=6)
            print(faces)
            for (x,y,w,h) in faces:
                roi=image_array[y:y+h,x:x+w]

                id_,conf=recognizer.predict(roi)
                if(conf<=30):
                    print(labels[id_],conf,file)
                else:
                    print(conf,file)    
                



