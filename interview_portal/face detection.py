import cv2

def face_detect():
    face_cascade=cv2.CascadeClassifier('D:\project\interview_portal\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
    video_show=cv2.VideoCapture(0)
    while True:
        ret,video_data=video_show.read()
        col=cv2.cvtColor(video_data,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(
            col,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30,30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        for (x,y,w,h) in faces:
            cv2.rectangle(video_data,(x,y),(x+w,y+h),(0,0,225),3)
        cv2.imshow('demo',video_data)
        if cv2.waitKey(10) == ord('w'):
            break
    video_show.release()

if __name__== "__main__":
    face_detect()