import numpy as np
import cv2
from skimage.feature import hog
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
    akaze = cv2.AKAZE_create(threshold=0.0001)
    Ntiles = len(tiles)
    ref_features = []
    for i in range(Ntiles):
        img = tiles[i]
        print(str(i+1)+'/'+str(Ntiles))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # computar keypoint y descritores
        kps, des = akaze.detectAndCompute(gray, mask=None)

        if len(kps) != 0:
            des = des[0]
        else:
            des = np.ones(61)*10**(20)

        ref_features.append( des )

    return np.array(ref_features)

def getFeauresORB(tiles):
    """
    Recibe lista con losetas y retorna caracteristicas con orb
    :params lista:
        cada item de la lista es una losetas en RGB
    :returns:
        shape: np.array (N de losetas, 1000)
    """
    orb = cv2.ORB_create(edgeThreshold = 31)

    Ntiles = len(tiles)
    ref_features = []
    for i in range(Ntiles):
        img = tiles[i]
        print(str(i+1)+'/'+str(Ntiles))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # computar keypoint y descritores
        kps, des = orb.detectAndCompute(gray, None)

        if len(kps) != 0:
            des = des[0]
        else:
            des = np.ones(32)*10**(20)

        ref_features.append( des )

    return np.array(ref_features)

def getFeauresSIFT(tiles):
    """
    Recibe lista con losetas y retorna caracteristicas con sift
    :params lista:
        cada item de la lista es una losetas en RGB
    :returns:
        shape: np.array (N de losetas, 1000)
    """
    sift = cv2.xfeatures2d.SIFT_create()

    Ntiles = len(tiles)
    ref_features = []
    for i in range(Ntiles):
        img = tiles[i]
        print(str(i+1)+'/'+str(Ntiles))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # computar keypoint y descritores
        kps, des = sift.detectAndCompute(gray, mask=None)

        if len(kps) != 0:
            des = des[0]
        else:
            des = np.ones(128)*10**(20)

        ref_features.append( des )

    return np.array(ref_features)

def getFeauresHOG(tiles):
    """
    Recibe lista con losetas y retorna caracteristicas con hog
    :params lista:
        cada item de la lista es una losetas en RGB
    :params int:
        0: para escala de grises, otro para rgb
    :returns:
        shape: np.array (N de losetas, 1000)
    """
    N = 800
    Ntiles = len(tiles)
    F = np.zeros((Ntiles,N))
    for i in range(Ntiles):
        print(str(i+1)+'/'+str(Ntiles))
        frame = tiles[i]
        frame = cv2.resize(frame,(140,140))
        fd, hog_image = hog(frame, orientations=8, pixels_per_cell=(14, 14),
                        cells_per_block=(1, 1), visualize=True, multichannel=True)
        F[i] = fd
    return F


def getFeatures(tiles, method):
    """
    recibe un str  y retorna caracteristicas en el metodo seleccionado
    CNN: para caracteristicas mediante red convolucional vgg16

    :params np.array img::
        mascara binaria
    :returns:
        shape: np.array (N de losetas, 1000)
    """
    methods = ['CNN','AKAZE','ORB','SIFT','HOG']
    method = methods[method]

    if method == 'CNN':
        model = VGG16(weights='imagenet',include_top = True)
        features = getFeauresCNN(tiles,model)
    elif method == 'AKAZE':
        features = getFeauresAKAZE(tiles)
    elif method == 'ORB':
        features = getFeauresORB(tiles)
    elif method == 'SIFT':
        features = getFeauresSIFT(tiles)
    elif method == 'HOG':
        features = getFeauresHOG(tiles)
    else:
        features = None

    return features
