import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Qt5Agg')


def show_all_checkerboards(rvecs, tvecs, checkerboard_size):

#     # Carregar os parâmetros de calibração
#     with np.load('calibration_data.npz') as data:
#         rvecs = data['rvecs']
#         tvecs = data['tvecs']

    # Criar gráfico 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotar a câmera na origem
    plot_camera(ax)

    # Adicionar rótulos aos eixos
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Título do gráfico
    ax.set_title('Posição dos Checkerboards Relativa à Câmera na Origem')

    # Plotar cada checkerboard
    for rvec, tvec in zip(rvecs, tvecs):
        plot_checkerboard(ax, rvec, tvec, checkerboard_size)

    # Ajustar os eixos para ter escalas iguais em x, y e z
    set_axes_equal(ax)

    # Mostrar gráfico interativo
    plt.show()


#_______________________________
# Get_calibration Sub Funcs
#_______________________________


def find_corners(images, checkerboard_size, square_size, criteria):
    
    # Preparar pontos do objeto
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size  # Multiplicar pelos tamanhos dos quadrados
        
    objpoints = []  # Pontos 3D no mundo real
    imgpoints = []  # Pontos 2D no plano da imagem

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
        else:
            print(f"Nenhum canto encontrado: {fname}")

    return objpoints, imgpoints, gray


def get_calibs(objpoints, imgpoints, gray):

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
    
    return ret, mtx, dist, rvecs, tvecs

#_______________________________
# print_calibrated_img Sub Funcs
#_______________________________

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

def print_calibrated_img(img, mtx, dist):

    # # Carregar os parâmetros de calibração
    # with np.load('calibration_data.npz') as data:
    #     mtx = data['mtx']
    #     dist = data['dist']

    # Carregar uma nova imagem
    #img = cv2.imread(r'C:\Users\RUI\ISEP\MEEC\2_ano\PROSE\Checkboards phone test\5.jpeg')
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

#_________________________________
# show_all_checkerboards Sub Funcs
#_________________________________

def plot_camera(ax):
    # Representar a câmera como um ponto
    camera_position = np.array([0, 0, 0])
    ax.scatter(camera_position[0], camera_position[1], camera_position[2], c='r', marker='o')
    
    # Definir os limites do quadrado por onde entra a luz na câmera
    square_size = [0.2, 0.15] # camera ratio 4/3
    camera_square = np.array([
        # [square_size[0] / 2, square_size[1] / 2, 0.1],
        # [-square_size[0] / 2, square_size[1] / 2, 0.1],
        # [-square_size[0] / 2, -square_size[1] / 2, 0.1],
        # [square_size[0] / 2, -square_size[1] / 2, 0.1]
        [square_size[0] / 2, 0.1, square_size[1] / 2],
        [-square_size[0] / 2, 0.1, square_size[1] / 2],
        [-square_size[0] / 2, 0.1, -square_size[1] / 2],
        [square_size[0] / 2, 0.1, -square_size[1] / 2]
    ])
    
    # Conectar o ponto da câmera aos limites do quadrado
    for corner in camera_square:
        ax.plot([camera_position[0], corner[0]], 
                [camera_position[1], corner[1]], 
                [camera_position[2], corner[2]], 'r-')
    
    # Conectar os cantos do quadrado para formar o limite
    ax.plot([camera_square[0][0], camera_square[1][0]], 
            [camera_square[0][1], camera_square[1][1]], 
            [camera_square[0][2], camera_square[1][2]], 'r-')
    ax.plot([camera_square[1][0], camera_square[2][0]], 
            [camera_square[1][1], camera_square[2][1]], 
            [camera_square[1][2], camera_square[2][2]], 'r-')
    ax.plot([camera_square[2][0], camera_square[3][0]], 
            [camera_square[2][1], camera_square[3][1]], 
            [camera_square[2][2], camera_square[3][2]], 'r-')
    ax.plot([camera_square[3][0], camera_square[0][0]], 
            [camera_square[3][1], camera_square[0][1]], 
            [camera_square[3][2], camera_square[0][2]], 'r-')


def plot_checkerboard(ax, rvec, tvec, checkerboard_size, square_size=0.030):
    # Converter rvec para matriz de rotação
    rot_matrix = cv2.Rodrigues(rvec)[0]
    
    # Definir os quatro cantos do checkerboard no sistema de coordenadas do mundo real
    checkerboard_corners = np.array([
        # [0, 0, 0],
        # [checkerboard_size[0] * square_size, 0, 0],
        # [checkerboard_size[0] * square_size, checkerboard_size[1] * square_size, 0],
        # [0, checkerboard_size[1] * square_size, 0]
        [checkerboard_size[0] * square_size / 2, checkerboard_size[1] * square_size / 2, 0],
        [-checkerboard_size[0] * square_size / 2, checkerboard_size[1] * square_size / 2, 0],
        [-checkerboard_size[0] * square_size / 2, -checkerboard_size[1] * square_size / 2, 0],
        [checkerboard_size[0] * square_size / 2, -checkerboard_size[1] * square_size / 2, 0]
        # [checkerboard_size[0] * square_size / 2, 0, checkerboard_size[1] * square_size / 2],
        # [-checkerboard_size[0] * square_size / 2, 0, checkerboard_size[1] * square_size / 2],
        # [-checkerboard_size[0] * square_size / 2, 0, -checkerboard_size[1] * square_size / 2],
        # [checkerboard_size[0] * square_size / 2, 0, -checkerboard_size[1] * square_size / 2]
    ])
    
    # Transformar os cantos do checkerboard para a posição e orientação da câmera
    transformed_corners = np.dot(rot_matrix, checkerboard_corners.T).T + tvec.T

    # Corrigir as posições para y ser a distância e z a altura 
    transformed_corners = transformed_corners[:, [0, 2, 1]]
    transformed_corners[:, 2] *= -1
    
    # Plotar o retângulo representando o checkerboard
    ax.plot([transformed_corners[0][0], transformed_corners[1][0]], 
            [transformed_corners[0][1], transformed_corners[1][1]], 
            [transformed_corners[0][2], transformed_corners[1][2]], 'k-')
    ax.plot([transformed_corners[1][0], transformed_corners[2][0]], 
            [transformed_corners[1][1], transformed_corners[2][1]], 
            [transformed_corners[1][2], transformed_corners[2][2]], 'k-')
    ax.plot([transformed_corners[2][0], transformed_corners[3][0]], 
            [transformed_corners[2][1], transformed_corners[3][1]], 
            [transformed_corners[2][2], transformed_corners[3][2]], 'k-')
    ax.plot([transformed_corners[3][0], transformed_corners[0][0]], 
            [transformed_corners[3][1], transformed_corners[0][1]], 
            [transformed_corners[3][2], transformed_corners[0][2]], 'k-')


def set_axes_equal(ax):
    '''Define os limites dos eixos para ter escalas iguais em x, y e z.'''
    
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max(x_range, y_range, z_range)

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
    ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
    ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])
