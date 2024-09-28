import cv2
import os
import time
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from mpl_toolkits.axes_grid1 import ImageGrid

SQUARE_SIZE = 30
BOARD_SIZE = (11,7)

LEFT_PATH = '/Users/sivaprasanth/Documents/Computer Vision/Computer-Vision/img/chessboard/leftcamera'
RIGHT_PATH = '/Users/sivaprasanth/Documents/Computer Vision/Computer-Vision/img/chessboard/rightcamera'

print('We have {} Images from the left camera'.format(len(os.listdir(LEFT_PATH))))
print('and {} Images from the right camera.'.format(len(os.listdir(RIGHT_PATH))))

print('Before: {}, {}, {}, ...'.format(os.listdir(LEFT_PATH)[0], os.listdir(LEFT_PATH)[1], os.listdir(LEFT_PATH)[2]))

def SortImageNames(path):
    imagelist = sorted(os.listdir(path))
    lengths = []
    for name in imagelist:
        lengths.append(len(name))
    lengths = sorted(list(set(lengths)))
    ImageNames, ImageNamesRaw = [], []
    for l in lengths:
        for name in imagelist:
            if len(name) == l:
                ImageNames.append(os.path.join(path, name))
                ImageNamesRaw.append(name)
    return ImageNames
                
Left_Paths = SortImageNames(LEFT_PATH)
Right_Paths = SortImageNames(RIGHT_PATH)

print('After: {}, {}, {}, ...'.format(os.path.basename(Left_Paths[0]), os.path.basename(Left_Paths[1]), os.path.basename(Left_Paths[2])))

example_image = cv2.imread(Left_Paths[5])
example_image = cv2.cvtColor(example_image, cv2.COLOR_BGR2GRAY)

ret, _ = cv2.findChessboardCorners(example_image, BOARD_SIZE)
if ret:
    print('Board Size {} is correct.'.format(BOARD_SIZE))
else:
    print('[ERROR] the Board Size is not correct!')
    BOARD_SIZE = (0,0)
    
objpoints = np.zeros((BOARD_SIZE[0]*BOARD_SIZE[1], 3), np.float32)
objpoints[:,:2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1,2)
objpoints *= SQUARE_SIZE

def GenerateImagepoints(paths):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    imgpoints = []
    for name in paths:
        img = cv2.imread(name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners1 = cv2.findChessboardCorners(img, BOARD_SIZE)
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners1, (4,4), (-1,-1), criteria)
            imgpoints.append(corners2)
    return imgpoints

Left_imgpoints = GenerateImagepoints(Left_Paths)
Right_imgpoints = GenerateImagepoints(Right_Paths)

def DisplayImagePoints(path, imgpoints):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.drawChessboardCorners(img, BOARD_SIZE, imgpoints, True)
    return img
    
example_image_left = DisplayImagePoints(Left_Paths[15], Left_imgpoints[15])
example_image_right = DisplayImagePoints(Right_Paths[15], Right_imgpoints[15])

fig = plt.figure(figsize=(20,20))
grid = ImageGrid(fig, 111, nrows_ncols=(1, 2), axes_pad=0.1)

for ax, im in zip(grid, [example_image_left, example_image_right]):
    ax.imshow(im)
    ax.axis('off')