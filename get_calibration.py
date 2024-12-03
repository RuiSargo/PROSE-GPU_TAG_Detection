import cv2
import numpy as np
import glob

from my_funcs import *

# pip install -r .\requirements.txt

# x vertices horizontais internos; y vertices verticais internos
checkerboard_size = (7,7)#(8, 6) # (x, y)

square_size = 0.050  # 30mm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# Get Calibration

# For Raw Images
# raw_path = r'C:\Users\RUI\ISEP\MEEC\2_ano\PROSE\Checkboards_phone_test\Test5\*.cr2'
# scale_percent = 25 # 25 % 
# processed_path = process_raw_images(raw_path, scale_percent)
# images = glob.glob(processed_path)
# print( len(images) )
# After raw processed to jpg
path = r'C:\Users\RUI\ISEP\MEEC\2_ano\PROSE\Checkboards_phone_test\Test5\Processed_raws\Best\*.jpg'
images = glob.glob(path)

data_info = find_corners(images, checkerboard_size, square_size, criteria)
ret, mtx, dist, rvecs, tvecs = get_calibs(*data_info)

# Get intrinsic values after calibration
###
with np.load('calibration_data.npz') as data:
    mtx = data['mtx']
    dist = data['dist']
    rvecs = data['rvecs']
    tvecs = data['tvecs']
###

# Visualize distortion Correction
img = cv2.imread(path[:-6]+r'\IMG_5077.jpg')
print_calibrated_img(img, mtx, dist)

# Show all detected checkerboards in relation to camara 
show_all_checkerboards(rvecs, tvecs, checkerboard_size)