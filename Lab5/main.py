import cv2
import numpy as np
def preprocess(file, kersize: int):
    img = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(img, (kersize, kersize), 0)


cap = cv2.VideoCapture('./Images/boo.mov')
#cap = cv2.VideoCapture(0)

kernel = 5
minarea =100
threshold = 25
ret, frame = cap.read()


w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter("Lab5output.mp4", fourcc, fps, (w, h))

img = preprocess(frame, kernel)
while True:

    ret2, frame2 = cap.read()
    cv2.imshow('boo', frame2)
    img_copy = frame2
    img2 = preprocess(frame2, kernel)

    imgdif = cv2.absdiff(img, img2)
    cv2.imshow('boo2', imgdif)
    ret3, imgdif = cv2.threshold(imgdif, threshold, 255, 0)
    #cont = cv2.findContours()
    cv2.imshow('boo3', imgdif)

    contours, hierarchy = cv2.findContours(imgdif, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    has_motion = False
    for cnt in contours:
        contArea = cv2.contourArea(cnt)
        if contArea >= minarea:
            print(contArea)
            has_motion = True
            break

    if has_motion:
        print("Don't move, or it will see you...")
        video_writer.write(img_copy)
    frame_prev = frame
    img = img2
    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
