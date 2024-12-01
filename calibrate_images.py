import cv2
import numpy as np

# Carregar os parâmetros de calibração
with np.load('calibration_data.npz') as data:
    mtx = data['mtx']
    dist = data['dist']

# Função para corrigir a distorção de uma imagem
def undistort_image(img, mtx, dist):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    
    # Corrigir a distorção
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    
    # Recortar a imagem
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst

def resizable_imshow(name,img):
    # Criar uma janela redimensionável
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 800, 600)
    cv2.imshow(name, img)

# Carregar uma nova imagem
img = cv2.imread(r'C:\Users\RUI\ISEP\MEEC\2_ano\PROSE\Checkboards phone test\5.jpeg')
if img is not None:
    undistorted_img = undistort_image(img, mtx, dist)
    
    invrt_img = cv2.bitwise_not(img)
    invrt_img = cv2.resize(invrt_img, (undistorted_img.shape[1], undistorted_img.shape[0]))
    blended_img = cv2.addWeighted(undistorted_img, 0.5, invrt_img, 1-0.5, 0)

    #resizable_imshow('Imagem Original',img)
    #resizable_imshow('Imagem Undistorted',undistorted_img)
    resizable_imshow('Dif original - undistorted', blended_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Erro ao carregar a imagem.")