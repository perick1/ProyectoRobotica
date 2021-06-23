import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
import cv2

def getDistanceMatrix(XA,XB,metric):
    """
    Recibe 2 matrices con características y retorna distancia metric de todas
    las combinaciones.
    :params 2 np.array features::
        shape: (N de losetas, 1000)
    :params metrica::
        entre ['euclidean','cityblock','cosine','correlation','chebyshev','canberra','braycurtis']
    :returns:
        shape: Pandas dataframe (len(XA),len(XB))
    """
    distances = distance.cdist( XA = XA.astype(float),
                                XB = XB.astype(float),
                                metric = metric)
    return pd.DataFrame(data=distances, index = np.arange(len(XA)), columns = np.arange(len(XB)))

def getOrder(distances):
    """
    Recibe matriz de distancias y retorna distancia metric de todas
    las combinaciones.
    :params pd.DataFrame::
        shape: (len(XA),len(XB))
    :returns:
        shape: 2 Pandas dataframe [distances.shape,distances.shape]
        con distancias en orden y los nombres correspondientes
    """
    D   = distances.values.astype(float)
    col = distances.columns.values
    idx = distances.index
    X   = np.zeros(distances.shape,dtype=float)
    Ns  = np.zeros(distances.shape,dtype=object)

    for k in range(len(distances)):
        d = D[k]
        ord_idx = np.argsort(d)
        d_n = d[ord_idx]
        names_n = col[ord_idx]
        X[k] = d_n
        Ns[k] = names_n
    return [pd.DataFrame(data=X , index = idx), pd.DataFrame(data=Ns, index = idx)]

def Examples(tiles,idx,Ns,Ds):
    fig, axs = plt.subplots(nrows=3,ncols=5,figsize=(8, 5))
    axx = axs.flat
    ax0 = axx[2]
    axx[0].set_axis_off()
    axx[1].set_axis_off()
    tile_original = tiles[idx]
    ax0.imshow(tile_original)
    ax0.set_title('WOLAS')
    ax0.set_axis_off()
    axx[3].set_axis_off()
    axx[4].set_axis_off()
    m = 255

    n = Ns.loc[idx].values
    d = Ds.loc[idx].values
    for i in range(10):
        ax = axx[i+5]
        img = tiles[n[i]]
        ax.imshow(img)
        print(d[i])
        #ax.set_title(str(n[i]))
        ax.text(3, 3, str(i+1),color='white',weight='bold',size=18,ha='left',va='top',
            bbox=dict(boxstyle="round",
                       ec=(35/m, 167/m, 168/m),
                       fc=(25/m, 135/m, 136/m),))
        ax.set_axis_off()
    fig.tight_layout()
    plt.show()
