import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


eye_cascade = cv2.CascadeClassifier('eye_haar_cascade.xml')





def find_iris(img_path):
    img = cv2.imread(img_path)
    hght = 500
    wdth = int(500*(img.shape[0]/img.shape[1]))
    resized_image = cv2.resize(img, (hght, wdth))
    if wdth < hght:
        center = int(hght/2)
        resized_image = resized_image[0:wdth,int(center-center/2):int(center+center/2),:]
    elif hght > wdth:
        center = int(wdth/2)
        resized_image = resized_image[int(center-center/2):int(center+center/2),0:hght,:]
    
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray)
    max_area = 0
    if len(eyes) > 0:
        for (ex, ey, ew, eh) in eyes:
            if ew*eh >= max_area:
                #cv2.rectangle(eyes, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                eroi = gray[ey:ey+eh, ex:ex+ew]
                max_area = ew*eh
    else:
          eroi = gray
    img = cv2.medianBlur(eroi, 5)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 50, param1=50, param2=30, minRadius=0, maxRadius=70)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        circle = max(circles[0], key=lambda x: x[2])
        print("Circles = " , circles)
        #for i in circles[0, :]:
            #cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw the center of the circle.
            #cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)  
        cv2.circle(cimg, (circle[0], circle[1]), circle[2], (255, 0, 0), 3 )  
                    
        plt.imshow(cimg)
        plt.show()
    else:
        print("Iris not found")
    




for file  in os.listdir(r"/home/bitnami/ctpro_april24/ctpro/static/drishti/images"):
    print(file)
    find_iris(r"/home/bitnami/ctpro_april24/ctpro/static/drishti/images/" + file)