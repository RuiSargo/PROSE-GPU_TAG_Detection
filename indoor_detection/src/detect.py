import tensorflow as tf
import numpy as np

def load_model(model_path='../models/indoor_model.h5'):
    """
    Carrega o modelo treinado.
    """
    return tf.keras.models.load_model(model_path)

def predict(model, image_path, image_size=(224, 224)):
    """
    Faz uma previsão para uma imagem fornecida.
    """
    # Carregar e processar a imagem
    image = tf.keras.utils.load_img(image_path, target_size=image_size)
    image_array = tf.keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)  # Adiciona a dimensão do batch

    # Fazer a previsão
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=-1)  # Classe com maior probabilidade
    return predicted_class

if __name__ == "__main__":
    model = load_model()
    image_path = '../converted_images/4x4/4x4_1000-0.png'  # Atualize com o caminho correto
    predicted_class = predict(model, image_path)
    print(f"Classe prevista: {predicted_class}")
