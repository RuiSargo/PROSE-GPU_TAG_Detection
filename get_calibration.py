import cv2
import numpy as np
import glob

from my_funcs import *

# x vertices horizontais internos; y vertices verticais internos
checkerboard_size = (8, 6) # (x, y)
square_size = 0.030  # 30mm
criteria = (cv2.TERM_CRITERIAs_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Get Calibration
#images = glob.glob(r'C:\Users\RUI\ISEP\MEEC\2_ano\PROSE\Checkboards phone test\Test2\*.jpeg')
images = glob.glob(r'C:\Users\RUI\ISEP\MEEC\2_ano\PROSE\Checkboards phone test\*.jpeg')
objpoints, imgpoints, gray = find_corners(images, checkerboard_size, square_size, criteria)
ret, mtx, dist, rvecs, tvecs = get_calibs(objpoints, imgpoints, gray)


###
# with np.load('calibration_data.npz') as data:
#     mtx = data['mtx']
#     dist = data['dist']
#     rvecs = data['rvecs']
#     tvecs = data['tvecs']
###

# Visualize distortion Correction
img = cv2.imread(r'C:\Users\RUI\ISEP\MEEC\2_ano\PROSE\Checkboards phone test\10.jpeg')
print_calibrated_img(img, mtx, dist)

# Show all detected checkerboards in relation to camara 
show_all_checkerboards(rvecs, tvecs, checkerboard_size)