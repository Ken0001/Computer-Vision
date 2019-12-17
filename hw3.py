import cv2
import numpy as np
import sys

def getMask(roi, img2):
    # roi is background region, img2 is foreground
    # 參考 https://www.itread01.com/content/1548463705.html
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 254, 255, cv2.THRESH_BINARY) 
    mask_inv = cv2.bitwise_not(mask)
    #cv.imshow('mask',mask_inv)
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask) 
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask_inv) 
    dst = cv2.add(img1_bg,img2_fg)
    return dst

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
mustache = cv2.imread("mustache.png")
hat = cv2.imread("hat.png") # 304x277
#cap = cv2.VideoCapture('wiiplay.mp4')
cap = cv2.VideoCapture(0)
scaling_factor = 0.5
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Face, mouth, eyes
    face_rects = face_cascade.detectMultiScale(frame, scaleFactor=1.5, minNeighbors=3)
    mouth_rects = mouth_cascade.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=4)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    #### Not used 
    # frame[y:y+h, x:x+w, :] => Face location
    # Select the region in the background where we want to add the image and add the images using cv2.addWeighted()
    # Ex: added_image = cv2.addWeighted(background[150:250,150:250,:],alpha,foreground[0:100,0:100,:],1-alpha,0)
    # cv2.addWeighted(frame[y:y+h, x:x+w, :], alpha, hat, 1-alpha, 0)

    for (x,y,w,h) in face_rects:
        #cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,0), 1)
        sw = (w/304)
        sh = int(277*sw)
        hat = cv2.resize(hat, (w, sh), interpolation=cv2.INTER_CUBIC)
        fx = x
        fX = x+w
        fy = y-int(0.6*h)
        fY = y+sh-int(0.6*h)
        roi = frame[fy:fY, fx:fX]
        if(fX >= 320 or fx <= 0 or fY>=240 or fy <= 0):
            break
        #print("ROI",roi.shape,",HAT",mus.shape," ALL",(x,y,w,h))
        dst = getMask(roi, hat)
        frame[y-int(0.6*h):y+sh-int(0.6*h), x:x+w] = dst
    
    for (x,y,w,h) in mouth_rects:
        y = int(y - 0.15*h)
        # nx nX, ny nY = 鬍子的寬, 高
        nx = x-int(0.5*w)
        nX = x+int(0.5*w)+w
        ny = y-int(0.5*h)
        nY = y-int(0.5*h)+int(1.35*h)
        #cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 1)
        mus = cv2.resize(mustache, ((2*w), int(1.35*h)), interpolation=cv2.INTER_CUBIC)
        roi = frame[ny:nY, nx:nX]
        # 防止超出邊界
        if(nX >= 320 or nx <= 0 or nY>=240 or ny <= 0):
            break
        if(mus.shape[1] > roi.shape[1]):
            mus = cv2.resize(mustache, ((2*w)-1, int(1.35*h)), interpolation=cv2.INTER_CUBIC)
        if(mus.shape[1] < roi.shape[1]):
            mus = cv2.resize(mustache, ((2*w)+1, int(1.35*h)), interpolation=cv2.INTER_CUBIC)
        #print("ROI",roi.shape,",MUS",mus.shape," A:",(x,y,w,h))
        dst = getMask(roi, mus)
        frame[ny:nY, nx:nX] = dst
        break

    for (x,y,w,h) in faces:
        #img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            (ecx, ecy)=(ex+int(ew/2), ey+int(eh/2))
            cv2.circle(roi_color,(ecx,ecy),int(ew/2.2),(0,0,0),3)
    cv2.imshow('My Detector', frame)
    c = cv2.waitKey(1)
    if c == 27:
        break
cap.release()
cv2.destroyAllWindows()
