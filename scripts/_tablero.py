import numpy as np
import matplotlib.pyplot as plt
import cv2

#---------------------------------------------------------
def getEdges(img):
    """
    Crea máscara y obtiene bordes dentro de un pantallazo

    :params np.array img:
        pantallazo

    :returns:
        lista de 2 elementos
        mask: mascara binaria
        edges: bordes dentro de img
    """

    img = cv2.GaussianBlur(img, (3, 3), 0)
    Img_hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)

    lower_hsv = np.array([0,130,135])
    upper_hsv = np.array([255,255,190])
    mask = cv2.inRange(Img_hsv,lower_hsv,upper_hsv)
    mask = np.uint8(mask)

    kernel = np.ones((7,7),np.float32)/25
    dst = cv2.filter2D(mask,-1,kernel)
    thresh1 = cv2.inRange(dst, 100, 255)
    dst = cv2.filter2D(thresh1,-1,kernel)
    thresh1 = cv2.inRange(dst, 100, 255)
    dst = cv2.filter2D(thresh1,-1,kernel)
    thresh1 = cv2.inRange(dst, 100, 255)
    edges = cv2.Canny(thresh1,threshold1=100,threshold2=255)

    mask = np.uint8(thresh1)

    return [mask,edges]

#---------------------------------------------------------
def getBiggestContour(mask):
    """
    Recibe una mascara, identifica objetos dentro y entrega
    el contorno más grande encotrado

    :params np.array mask:
        mascara binaria

    :returns:
        big: numpy array con lar coordenadas del borde del mapa
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    N_cont = len(contours)
    bigger_area = 0
    for i in range(N_cont):
        area = cv2.contourArea(contours[i])
        if area > bigger_area:
            bigger_area = area
            idx = i
    big_contour = contours[idx]
    big = big_contour[:,0,:]

    return big

#---------------------------------------------------------
def getBoard(img,contour):
    """
    Recibe pantallazo y entrega el recorte del tablero

    :params np.array img:
        pantallazo
    :params np.array contour:
        coordenadas de los bordes del tablero

    :returns:
        img_cutted: numpy array, imagen del tablero aislado
    """

    xmin,ymin = contour.min(axis=0) + 6
    xmax,ymax = contour.max(axis=0) - 6
    img_cutted = img[ymin:ymax,xmin:xmax,:]

    return img_cutted

#---------------------------------------------------------
def plotBoard(img):
    """
    Recibe pantallazo y entrega la misma imagen pero rearcado donde
    está el tablero

    :params np.array img:
        pantallazo

    :returns:
        img_tab: numpy array, imagen pantallazo con el tablero marcado
    """

    pass
