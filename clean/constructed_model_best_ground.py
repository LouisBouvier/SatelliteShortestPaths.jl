import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from config import PATH_DATA
from architecture_model import MODEL_V0

AUTOTUNE = tf.data.AUTOTUNE

def creation_dataset(im_size:tuple=(64,64), batch_size:int=32)->tuple:
    """
    
    """

    train_ds = tf.keras.utils.image_dataset_from_directory(
        PATH_DATA,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=im_size,
        batch_size=batch_size)
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        PATH_DATA,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=im_size,
        batch_size=batch_size)
    
    return train_ds, val_ds

def preprocessing(train_ds, val_ds)->tuple:
    """
    
    """

    normalization_layer = tf.keras.layers.Rescaling(1./255)
    cropper_layer = tf.keras.layers.Cropping2D(cropping=((16, 16)))

    cropped_ds = train_ds.map(lambda x, y: (cropper_layer(x), y))
    normalized_ds_train = cropped_ds.map(lambda x, y: (normalization_layer(x), y))

    cropped_ds = val_ds.map(lambda x, y: (cropper_layer(x), y))
    normalized_ds_valid = cropped_ds.map(lambda x, y: (normalization_layer(x), y))

    train_ds = normalized_ds_train.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = normalized_ds_valid.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds


if __name__ == "__main__":

    train_ds, val_ds = creation_dataset()
    train_ds, val_ds = preprocessing(train_ds, val_ds)

    MODEL_V0.compile(optimizer='adam',
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
    
    earlyStopping = EarlyStopping(monitor='val_accuracy', patience=10, verbose=0, mode='auto')
    mcp_save = ModelCheckpoint("ground_finder.keras", save_best_only=True, monitor='val_accuracy', mode='auto')

    # Train the model
    history = MODEL_V0.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    callbacks=[earlyStopping, mcp_save]
    )

    MODEL_V0.save('ground_finder.keras')