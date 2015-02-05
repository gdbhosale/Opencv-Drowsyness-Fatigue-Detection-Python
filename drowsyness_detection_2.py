import numpy as np
import cv2
from matplotlib import pyplot as plt

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades2/haarcascade_eye.xml')

simulate_real_time = "false"

if(simulate_real_time == "true"):
    cap = cv2.VideoCapture(0)

def detect_and_draw(img, gray):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img, (x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        max_eyes = 2
        cnt_eye = 0
        for (ex,ey,ew,eh) in eyes:
            if(cnt_eye == max_eyes):
                break;
            
            image_name = 'Eye_' + str(cnt_eye)
            print image_name
            
            #change dimentionas
            ex = ex + (ew/6)
            ew = ew - (ew/6)
            ey = ey + (eh/3)
            eh = eh/3
            
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)
            
            if(simulate_real_time == "false"):

                roi_eye_gray = roi_gray[ey:ey+eh, ex:ex+ew]
                roi_eye_color = roi_color[ey:ey+eh, ex:ex+ew]

                # create & normalize histogram ---------
                hist = cv2.calcHist([roi_eye_gray], [0],None, [256], [0,256])
                histn = []
                max_val = 0
                for i in hist:
                    value = int(i[0])
                    histn.append(value)
                    if(value > max_val):
                        max_val = value
                for index, value in enumerate(histn):
                    histn[index] = ((value * 256) / max_val)
                print histn
                # normalize histogram ends ---------

                # Plot Histogram
                plt.subplot(2,3,((cnt_eye*3)+1)),plt.hist(roi_eye_gray.ravel(), 256, [0,256])
                plt.title(image_name+' Hist')

                # Slice
                roi_eye_gray2 = roi_eye_gray.copy()
                #roi_eye_gray2 = cv2.threshold(roi_eye_gray2, 50, 255, cv2.THRESH_TOZERO)
                #roi_eye_gray2 = cv2.adaptiveThreshold(roi_eye_gray2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 10)

                threshold = 65
                for i in range(0, roi_eye_gray2.shape[0]):
                    for j in range(0, roi_eye_gray2.shape[1]):
                        pixel_value = roi_eye_gray2[i, j]
                        if(pixel_value >= threshold):
                            roi_eye_gray2[i, j] = 255
                        else:
                            roi_eye_gray2[i, j] = 0

                #cv2.imshow(image_name, roi_eye_gray2)


                # Plot Eye Images
                plt.subplot(2,3,((cnt_eye*3)+2)),plt.imshow(roi_eye_color, 'gray')
                plt.title(image_name+' Image Threshold')

                # Plot Eye Images after threshold
                plt.subplot(2,3,((cnt_eye*3)+3)),plt.imshow(roi_eye_gray2, 'gray')
                plt.title(image_name+' Image')

            cnt_eye = cnt_eye + 1
            
    cv2.imshow('frame', img)
    if(simulate_real_time == "false"):
        plt.show()
    
    

if __name__ == '__main__':
    if(simulate_real_time == "true"):
        while(True):
            ret, frame = cap.read()
            frame = cv2.resize(frame, (600, 350))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detect_and_draw(frame, gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        # Capture frame-by-frame
        frame = cv2.imread('face_img.jpg')

        # Resize Image
        frame = cv2.resize(frame, (600, 350))

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detect_and_draw(frame, gray)
        #cv2.waitKey(0)

        # Display the resulting frame
        #cv2.imshow('frame',gray)


    # When everything done, release the capture
    cv2.destroyAllWindows()