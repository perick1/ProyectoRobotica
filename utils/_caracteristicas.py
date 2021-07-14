import numpy as np
import cv2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

def getFeauresCNN(tiles,model):
    """
    Recibe lista con losetas y retorna caracteristicas con VGG16
    :params lista:
        cada item de la lista es una losetas en RGB
    :returns:
        shape: np.array (N de losetas, 1000)
    """
    #global model
    #model = VGG16(weights='imagenet',include_top = True)
    N = 1000
    Ntiles = len(tiles)
    F = np.zeros((Ntiles,N))
    for i in range(Ntiles):
        print(str(i+1)+'/'+str(Ntiles))
        tile = cv2.resize(tiles[i], (224,224), interpolation = cv2.INTER_AREA)
        x = np.expand_dims(tile, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        n = features.flatten()
        F[i] = n
    return F

def getFeauresAKAZE(tiles):
    """
    Recibe lista con losetas y retorna caracteristicas con akaze
    :params lista:
        cada item de la lista es una losetas en RGB
    :returns:
        shape: np.array (N de losetas, 1000)
    """
    r =  'no implementado aun'
    return r

def getFeatures(tiles, method):
    """
    recibe un str  y retorna caracteristicas en el metodo seleccionado
    CNN: para caracteristicas mediante red convolucional vgg16

    :params np.array img::
        mascara binaria
    :returns:
        shape: np.array (N de losetas, 1000)
    """
    if method == 'CNN':
        model = VGG16(weights='imagenet',include_top = True)
        features = getFeauresCNN(tiles,model)
    elif method == 'SIFT':
        features = 'no implementado aun'
    else:
        features = None

    return features
