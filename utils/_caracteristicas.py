import numpy as np
import cv2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

def getFeauresCNN(tiles):
    """
    Recibe lista con losetas y retorna caracteristicas con VGG16
    :params np.array img::
        mascara binaria
    :returns:
        shape: np.array (N de losetas, 1000)
    """
    model = VGG16(weights='imagenet',include_top = True)
    N = 1000
    Ntiles = len(tiles)
    F = np.zeros((Ntiles,N))
    for i in range(Ntiles):
        tile = cv2.resize(tiles[i], (224,224), interpolation = cv2.INTER_AREA)
        x = np.expand_dims(tile, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        n = features.flatten()
        F[i] = n
    return F
