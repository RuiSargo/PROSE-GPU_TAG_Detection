import cv2
        
#_____________
def determine_image_type(image):
    if len(image.shape) == 2:
        return "Gray"
    elif len(image.shape) == 3:
        if image.shape[2] == 3:
            # Verificar se é RGB ou BGR
            b, g, r = cv2.split(image)
            if (image[:, :, 0] == b).all() and (image[:, :, 1] == g).all() and (image[:, :, 2] == r).all():
                return "BGR"
            else:
                return "RGB"
        else:
            return "Unknown"
    else:
        return "Unknown"

# Carregar uma imagem
image_path = r'C:\Users\RUI\ISEP\MEEC\2_ano\PROSE\Checkboards_phone_test\Test5\Processed_raws\IMG_5092.jpg'
image = cv2.imread(image_path)

# Determinar o tipo de imagem
image_type = determine_image_type(image)
print(f'Tipo da imagem: {image_type}')
#_____________