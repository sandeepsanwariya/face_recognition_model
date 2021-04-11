import cv2
import numpy as np
face_classifier =cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def get_face(img):
    faces =face_classifier.detectMultiScale(img,1.3,5)
    if faces is ():
        return None
    for (x,y,w,h) in faces:
        x=x-10
        y=y-10
        croped_face= img[y:y+h+50,x:x+w+50]
    return croped_face


cap=cv2.VideoCapture(0)
cout=0
while True:
    ret,img=cap.read()
    if get_face(img) is not None:
        cout=cout+1
        face=cv2.resize(get_face(img),(400,400))
        file_path='./Dataset/test/'+str(cout)+'.jpg'
        cv2.imwrite(file_path,face)
        cv2.putText(face,str(cout),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow("face",face)

    else:
        print("face not found")
        pass
    if cv2.waitKey(5) & 0xFF == ord("q") or cout==100:
        break

cap.release()
cv2.destroyAllWindows()
