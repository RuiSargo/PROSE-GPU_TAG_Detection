import cv2
import numpy as np
import glob

# Defina o tamanho do checkerboard
# x vertices horizontais internos
# y vertices verticais internos
checkerboard_size = (8, 6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Defina a dimensão de cada quadrado do checkerboard (em unidades reais, por exemplo, metros ou milímetros)
square_size = 0.030  # 30mm

# Preparar pontos do objeto
objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
objp *= square_size  # Multiplicar pelos tamanhos dos quadrados

objpoints = []  # Pontos 3D no mundo real
imgpoints = []  # Pontos 2D no plano da imagem

# Carregar as imagens capturadas
#images = glob.glob(r'C:\Users\RUI\ISEP\MEEC\2_ano\PROSE\Checkboards phone test\Test2\*.jpeg')
images = glob.glob(r'C:\Users\RUI\ISEP\MEEC\2_ano\PROSE\Checkboards phone test\*.jpeg')

# Variável para armazenar a última imagem processada
last_img = None  

for fname in images:
    img = cv2.imread(fname)

    img = cv2.convertScaleAbs(img, alpha=1.25, beta=-135)

    # name='test'
    # cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(name, 800, 600)
    # cv2.imshow(name, img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if img is None:
        print(f"Erro ao carregar a imagem: {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Equalizar histograma para melhorar o contraste

    # name='test'
    # cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(name, 800, 600)
    # cv2.imshow(name, gray)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

    if ret:
        print(f"Cantos encontrados: {fname}")
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        cv2.drawChessboardCorners(img, checkerboard_size, corners2, ret)
        last_img = img  # Armazena a última imagem processada
    else:
        print(f"Nenhum canto encontrado: {fname}")

if len(objpoints) > 0 and len(imgpoints) > 0:
    # Calibrar a câmera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    #print(ret, '\n\n', mtx, '\n\n', dist, '\n\n', rvecs, '\n\n', tvecs, '\n')

    # Salvar os parâmetros de calibração
    np.savez('calibration_data.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    print("Calibração concluída e dados salvos.")
    print(len(objpoints)," imagens com checkerboards detetados!")
else:
    print("Erro: Não foram encontrados cantos suficientes para a calibração.")