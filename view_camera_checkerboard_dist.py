import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Configurar o backend interativo
import matplotlib
matplotlib.use('Qt5Agg')

checkerboard_size = (8, 6)

def plot_checkerboard(ax, rvec, tvec, checkerboard_size=(8, 6), square_size=0.030):
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
    ])
    
    # # Inverter os valores de y dos cantos do checkerboard
    # checkerboard_corners[:, 1] *= -1
    
    # Transformar os cantos do checkerboard para a posição e orientação da câmera
    transformed_corners = np.dot(rot_matrix, checkerboard_corners.T).T + tvec.T
    
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

def plot_camera(ax):
    # Representar a câmera como um ponto
    camera_position = np.array([0, 0, 0])
    ax.scatter(camera_position[0], camera_position[1], camera_position[2], c='r', marker='o')
    
    # Definir os limites do quadrado por onde entra a luz na câmera
    square_size = [0.2, 0.15]
    camera_square = np.array([
        [square_size[0] / 2, square_size[1] / 2, 0.1],
        [-square_size[0] / 2, square_size[1] / 2, 0.1],
        [-square_size[0] / 2, -square_size[1] / 2, 0.1],
        [square_size[0] / 2, -square_size[1] / 2, 0.1]
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

# Carregar os parâmetros de calibração
with np.load('calibration_data.npz') as data:
    rvecs = data['rvecs']
    tvecs = data['tvecs']


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
    plot_checkerboard(ax, rvec, tvec)

# Ajustar os eixos para ter escalas iguais em x, y e z
set_axes_equal(ax)

# Mostrar gráfico interativo
plt.show()