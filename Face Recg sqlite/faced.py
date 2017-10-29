from Tkinter import *
import os
import cv2
import numpy as np
from PIL import Image
import sqlite3

def trainer():
    recognizer=cv2.createLBPHFaceRecognizer();
    path='dataSet'

    def getImagesWithID(path):
        imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
        faces=[]
        IDs=[]
        for imagePath in imagePaths:
            faceImg=Image.open(imagePath).convert('L');
            faceNp=np.array(faceImg,'uint8')
            ID=int(os.path.split(imagePath)[-1].split('.')[1])
            faces.append(faceNp)
            IDs.append(ID)
            cv2.imshow("training",faceNp)
            cv2.waitKey(10)
        return np.array(IDs),faces

    Ids,faces=getImagesWithID(path)
    recognizer.train(faces,Ids)
    recognizer.save('recognizer/trainningData.yml')
    cv2.destroyAllWindows()

def Creater(id1,name1):
    faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cam=cv2.VideoCapture(0)

    def insertOrUpdate(Id,Name):
        conn=sqlite3.connect("FaceBase.db")
        cmd="SELECT * FROM people WHERE ID="+str(Id)
        cursor=conn.execute(cmd)
        isRecordExist=0
        for row in cursor:
            isRecordExist=1
        if(isRecordExist==1):
            cmd="UPDATE people SET Name='"+str(Name)+"'WHERE ID="+str(Id)
        else:
            cmd="INSERT INTO people(ID,Name) Values("+str(Id)+",'"+str(Name)+"')"
        conn.execute(cmd)
        conn.commit()
        conn.close()

    
    insertOrUpdate(id1,name1)
    sampleNum=0
    while(True):
        ret,img=cam.read()
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=faceDetect.detectMultiScale(gray,1.3,5)
        for(x,y,w,h) in faces:
            sampleNum=sampleNum+1
           
            cv2.imwrite("dataSet/User."+str(id1)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
           
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.waitKey(100)
        cv2.imshow("Face",img)
        cv2.waitKey(1)
        if(sampleNum>20):
            print("done")
            break
    cam.release()
    cv2.destroyAllWindows()
    trainer()
    
def detect():
    faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
    cam=cv2.VideoCapture(0);
    rec=cv2.createLBPHFaceRecognizer();
    rec.load("recognizer\\trainningData.yml")
    id=0
    path='dataSet'

    def getProfile(id):
        conn=sqlite3.connect("FaceBase.db")
        cmd="SELECT * FROM people WHERE ID="+str(id)
        cursor=conn.execute(cmd)
        profile=None
        for row in cursor:
            profile=row
        conn.close()
        return profile


    font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX,1,1,0,1,1)
    while(True):
        ret,img=cam.read();
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ##faces=faceCascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,minSize=(100,100),flags)
        faces=faceDetect.detectMultiScale(gray,1.3,5);
        for(x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            id,conf=rec.predict(gray[y:y+h,x:x+w])
        ##if(id==101):
          ##  id="Shantanu"
            profile=getProfile(id)
            if(profile!=None):
                cv2.cv.PutText(cv2.cv.fromarray(img),str(profile[1]),(x,y+h+30),font,255);
                cv2.cv.PutText(cv2.cv.fromarray(img),str(profile[2]),(x,y+h+60),font,255);
                cv2.cv.PutText(cv2.cv.fromarray(img),str(profile[3]),(x,y+h+90),font,255);
                ##cv2.cv.PutText(cv2.cv.fromarray(img),str(profile[4]),(x,y+h+120),font,255);
        cv2.imshow("Face",img);
        if(cv2.waitKey(1)==ord('q')):
            break;
    cam.release()
    cv2.destroyAllWindows()

top=Tk()
top.title("Face Detector")
L1 = Label(top, text="Name")
L1.pack(side = LEFT)
E1 = Entry(top,bd=5)
E1.pack()

L2 = Label(top, text="Id")
L2.pack( side = LEFT)
E2 = Entry(top, bd =5)
E2.pack()
def store():
    input_name=E1.get()
    input_id=E2.get()
    Creater(input_id,input_name)

def analyse():

    detect();
StoreButton = Button(top,text="Store",command=store)
AnalyseButton = Button(top,text="Analyse",command=analyse)
        
StoreButton.pack()
AnalyseButton.pack()

top.mainloop()
