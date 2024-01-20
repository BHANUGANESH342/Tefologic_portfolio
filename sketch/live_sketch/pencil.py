import cv2 as cv
import numpy as np



def sketch(image):
    img_greay = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    img_blur =cv.GaussianBlur(img_greay,(7,7),0)
    canny = cv.Canny(img_blur,60,70)
    ret,thres =cv.threshold(canny,90,120,cv.THRESH_BINARY)
    return thres

cv.destroyAllWindows()

#inalising web cam
cap= cv.VideoCapture(0)
while True :
    ret,frame = cap.read()
    cv.imshow("live demo bhanu" ,sketch(frame))
    if cv.waitKey(1)== 13:
        break
cv.destroyAllWindows()
cap.release()
cv.destroyAllWindows()
