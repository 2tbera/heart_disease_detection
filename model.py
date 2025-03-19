import numpy as np
import tensorflow as tf
import pandas as pd

df = pd.read_csv("HeartPredictonQuantuDataset.csv")  # Replace with your actual file path

data = df.values

x = np.array(data[:, :-1])
y = data[:, -1]

model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(x.shape[1],)),  # dynamically set input size
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ], name="my_model"
)

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy']
)

model.fit(
    x, y,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
)

model.save('my_model.keras')
