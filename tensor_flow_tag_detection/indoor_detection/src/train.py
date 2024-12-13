import tensorflow as tf
from tensorflow.keras import layers, models
from preprocess import load_dataset

def create_model(input_shape=(224, 224, 3), num_classes=5):
    """
    Cria um modelo básico de CNN.

    Args:
        input_shape (tuple): Dimensões da entrada.
        num_classes (int): Número de classes (baseado nas pastas do dataset).

    Returns:
        model: Modelo compilado.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')  # Softmax para classificação multi-classe
    ])
    return model

def train_model(data_dir, epochs=10, batch_size=32):
    """
    Treina o modelo com base no dataset fornecido.
    """
    # Carregar os datasets de treinamento e validação, e as classes
    train_ds, val_ds, class_names = load_dataset(data_dir, batch_size=batch_size)

    # Número de classes
    num_classes = len(class_names)

    # Criar o modelo
    model = create_model(num_classes=num_classes)

    # Compilar o modelo
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Treinar o modelo
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    # Salvar o modelo treinado
    model.save('../models/indoor_model.h5')
    print("Modelo treinado e salvo em '../models/indoor_model.h5'.")

if __name__ == "__main__":
    data_dir = '../converted_images'  # Atualize com o caminho correto
    train_model(data_dir)
