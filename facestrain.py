import os
from PIL import Image
import numpy as np
import cv2
import pickle
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
image_dir= os.path.join(BASE_DIR,"images")
face_cascade=cv2.CascadeClassifier('test1/cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade=cv2.CascadeClassifier('test1/cascades/data/haarcascade_eye.xml')

recognizer=cv2.face.LBPHFaceRecognizer_create()

current_id=0
label_ids={}
x_train=[]
y_labels=[]

for root,dirs,files in os.walk(image_dir):
    for file in files:
        if(file.endswith("jpg") or file.endswith("png") or file.endswith("JPG") or file.endswith("jpeg") or file.endswith("JPEG")):
            path=os.path.join(root,file)
            label=os.path.basename(os.path.dirname(path)).replace(" ","").lower()
            #print(label,path)
            if( not (label in label_ids)):
                label_ids[label]=current_id
                current_id+=1
            id_= label_ids[label]

            #print(label_ids)
                
                
            '''x_train.append(path)
            y_labels.append(label)'''
            pil_image=Image.open(path).convert("L")
            size=(300,300)
            final_image=pil_image.resize(size,Image.ANTIALIAS)
            #print(final_image)
            image_array=np.array(pil_image,"uint8")
            #print(image_array)
            faces=face_cascade.detectMultiScale(image=image_array,scaleFactor=1.2,minNeighbors=6)
            eyes=eye_cascade.detectMultiScale(image=image_array)
            print(faces,label)
            for (x,y,w,h) in faces:
                
                roi=image_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
                


with open("test1/labels.pickle","wb") as f:
    pickle.dump(label_ids,f)

recognizer.train(x_train,np.array(y_labels) )
recognizer.save("test1/trainer.yml")