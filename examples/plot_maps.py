import cv2
import pickle

with open('maps.pkl', 'rb') as f:
    maps = pickle.load(f)
    
# 4x4 grid of maps
import numpy as np

ims = []
    
for i in range(len(maps)):
    dict_ = maps[i]
    img = dict_['map']
    img_name = img + '.png'
    mat = cv2.imread(img_name)
    row = i // 2
    col = i % 2
    print(row, col)
    ims.append(mat)
    

row_1 = np.hstack(ims[:2])
row_2 = np.hstack(ims[2:])
large_map = np.vstack([row_1, row_2])
    
cv2.imwrite('large_map.png', large_map)
