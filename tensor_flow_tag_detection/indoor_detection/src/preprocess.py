import tensorflow as tf

def load_dataset(data_dir, image_size=(224, 224), batch_size=32, validation_split=0.2):
    """
    Carrega o dataset a partir de diretórios organizados por classes.
    
    Args:
        data_dir (str): Caminho para o diretório das imagens.
        image_size (tuple): Tamanho das imagens.
        batch_size (int): Tamanho do batch.
        validation_split (float): Proporção do dataset reservada para validação.

    Returns:
        train_ds, val_ds, class_names: Datasets de treinamento e validação, e nomes das classes.
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=123,
        image_size=image_size,
        batch_size=batch_size
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=123,
        image_size=image_size,
        batch_size=batch_size
    )

    class_names = train_ds.class_names  # Obter os nomes das classes
    
    # Melhorar o desempenho do pipeline
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names
