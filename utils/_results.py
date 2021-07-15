import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os, sys
import glob

def getNewBoard(match_info ,path_pieces):
    '''
    paths   = glob.glob(path_pieces+'*.png')
    N = len(paths)
    classes = []
    piecesize= 400
    pieces  = np.zeros((len(paths),piecesize,piecesize,3))
    for i in range(N):
        img = paths[i]
        name=os.path.basename(img)
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (piecesize,piecesize))#, interpolation = cv2.INTER_AREA)
        pieces[i,:,:,:] = img
        classes.append(name[-6:-4])

    classes = np.array(classes)
    '''
    piecesize= 200
    N = len(match_info)
    Nboard = np.zeros((N,piecesize,piecesize,3))
    classes = []

    #return pieces
    #Nboard = []
    for i in range(N):
        tile ,tile_match ,name ,clase ,sentido = match_info[i]
        p = path_pieces + 'c' + clase +'.png'
        #piece = (pieces[classes == clase])[0]
        piece = cv2.imread(p)
        piece = cv2.cvtColor(piece, cv2.COLOR_BGR2RGB)
        piece = cv2.resize(piece, (piecesize,piecesize))
        if sentido == '02':
            piece = cv2.rotate(piece, cv2.cv2.ROTATE_90_CLOCKWISE)
        elif sentido == '03':
            piece = cv2.rotate(piece, cv2.ROTATE_180)
        elif sentido == '04':
            piece = cv2.rotate(piece, cv2.ROTATE_90_COUNTERCLOCKWISE)
        Nboard[i,:,:,:] = piece
        classes.append(clase)
    #plt.imshow(piece.astype(float)/255)
    #plt.show(block=False)
    #Nboard = np.array(Nboard,dtype = object)
    return [Nboard,np.array(classes).astype(int)]

def plotBoard(NB ,dimentions):
    piecesize= 200
    weight , height = dimentions * piecesize
    Nb = np.zeros((height,weight,3),dtype='uint8')
    for i in range(dimentions[1]):
        for j in range(dimentions[0]):
            k = i * dimentions[0] + j
            piece = NB[k,:,:,:]
            Nb[i*piecesize:(i+1)*piecesize,j*piecesize:(j+1)*piecesize,:] = piece
    return Nb
