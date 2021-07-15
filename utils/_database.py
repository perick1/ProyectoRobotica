import numpy as np
import cv2
import os, sys
import glob


def getDatabase(path):
    """
    Recibe direccion de la base de datos y entrega listas con las imagenes
    base de datos y sus nombres
    :params str:
        path
    :returns:
        tiles: lista con losetas recortadas de tamaño 70 x 70 px
        names: lista con nombre de archivos de las imagenes en la base de datos
    """
    paths = glob.glob(path+'*.png')
    #paths = glob.glob('**/img/database/*.png', recursive=True)
    tiles = []
    names = []
    for img in paths:
        name=os.path.basename(img)
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tiles.append(img)
        names.append(name)
    return [np.array(tiles),np.array(names)]
