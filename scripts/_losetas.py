import numpy as np
import matplotlib.pyplot as plt
import cv2

#----------------------------------------------
def getMask(img_cut):
    img_hsv = cv2.cvtColor(img_cut,cv2.COLOR_RGB2HSV)

    lower_hsv = np.array([0,70,200])
    upper_hsv = np.array([35,150,255])
    mask1 = cv2.inRange(img_hsv,lower_hsv,upper_hsv)

    lower_hsv = np.array([0,70,100])
    upper_hsv = np.array([35,150,130])
    mask3 = cv2.inRange(Img_hsv,lower_hsv,upper_hsv)

    mask1 = (mask1+mask3 - 255) * 255

    return mask1

#----------------------------------------------
def getShape1(mask_cut):
    Ly = max(mask_cut.shape)
    Lx1 = min(mask_cut.shape)
    Lx2 = Ly - Lx1
    NoSquare = Lx2 > 50

    Lx = (Lx1,Lx2)
    Lxl = max (Lx)
    Lxc = min(Lx)

    if NoSquare:
        Ntotal = 30
        error1 = np.zeros((2,Ntotal))
        error2 = np.zeros((2,Ntotal))
        for i in range(Ntotal):
            leng = round (Lxc / (i + 1))
            entero1 = Ly // leng
            real1 = Ly / leng
            error1[0,i] = Ly % leng
            error1[1,i] =  error1[0,i] - entero1

            leng = round (Lxl / (i + 1))
            entero2 = Ly // leng
            real2 = Ly / leng
            error2[0,i] = Ly % leng
            error2[1,i] =  error2[0,i] - entero2
    return [error1,error2]
