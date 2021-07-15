import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def getNewBoard(match_info,path_pieces):
    paths   = glob.glob(path_pieces+'*.png')
    pieces  = []
    classes = []
    for img in paths:
        name=os.path.basename(img)
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pieces.append(img)
        clases.append(name[-6:-4])

    pieces  = np.array(pieces)
    classes = np.array(classes)

    Nboard = []
    for i in range(len(match_info)):
        tile ,tile_match ,name ,clase = match_info[i]
        piece = pieces[classes == clase]
        Nboard.append(piece)
    Nboard = np.array(Nboard)
    return Nboard
