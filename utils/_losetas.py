import numpy as np
import matplotlib.pyplot as plt
import cv2

#----------------------------------------------
def getMask(img_cut):
    """
    Recibe pantallazo solo del tablero y crea máscara
    :params np.array img:
        pantallazo
    :returns:
        mask: mascara binaria
    """
    img_hsv = cv2.cvtColor(img_cut,cv2.COLOR_RGB2HSV)

    lower_hsv = np.array([0,70,200])
    upper_hsv = np.array([35,150,255])
    mask1 = cv2.inRange(img_hsv,lower_hsv,upper_hsv)

    lower_hsv = np.array([0,70,100])
    upper_hsv = np.array([35,150,130])
    mask3 = cv2.inRange(img_hsv,lower_hsv,upper_hsv)

    mask1 = (mask1+mask3 - 255) * 255

    return mask1

#----------------------------------------------
def getShape(mask_cut):
    """
    Recibe mascara solo del tablero y retorna dimensiones del tablero
    :params np.array img::
        mascara binaria
    :returns:
        shape: np.array con el tamaño del tablero (losetas horizontales, losetas verticales)
    """
    pass

def getPieces(img_cut, dimentions):
    """
    Recibe pantallazo solo del tablero y crea máscara
    :params np.array img:
        pantallazo
    :returns:
        pieces: lista con losetas recortadas de tamaño 70 x 70 px
    """

    tile_size = 70
    new_dim = dimentions * tile_size
    new_dim = (new_dim[0],new_dim[1])
    resized = cv2.resize(img_cut, new_dim, interpolation = cv2.INTER_AREA)

    pieces = list()
    for i in range(dimentions[1]):
        for j in range(dimentions[0]):
            tile = resized[i*tile_size:(i+1)*tile_size,
                           j*tile_size:(j+1)*tile_size,:]
            pieces.append(tile)
    return pieces
