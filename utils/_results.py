import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os, sys
import glob

def getNewBoard(match_info ,path_pieces):
    paths   = glob.glob(path_pieces+'*.png')
    pieces  = []
    classes = []
    piecesize= 400
    for img in paths:
        name=os.path.basename(img)
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (piecesize,piecesize))#, interpolation = cv2.INTER_AREA)
        pieces.append(img)
        classes.append(name[-6:-4])

    pieces  = np.array(pieces,dtype = object)
    classes = np.array(classes)

    Nboard = []
    for i in range(len(match_info)):
        tile ,tile_match ,name ,clase = match_info[i]
        piece = pieces[classes == clase]
        Nboard.append(piece)
    Nboard = np.array(Nboard,dtype = object)
    return Nboard

def plotNewBoard(NB ,dimentions):
    piecesize = 400
    weight , height = dimentions * piecesize
    Nb = np.ones((height,weight,3))
    for i in range(dimentions[1]):
        for j in range(dimentions[0]):
            k = i * dimentions[1] + j
            Nb[i*piecesize:(i+1)*piecesize,j*piecesize:(j+1)*piecesize,:] = (NB[k])[0]
    return Nb
