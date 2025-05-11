import tensorflow as tf
import numpy as np

def create_ff_model(sequence_length):
    """Crea un modelo feedforward simple para la predicción de secuencias."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(sequence_length,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1)  # Una única salida para la predicción
    ])
    return model

def compile_model(model, learning_rate=0.001):
    """Compila el modelo con el optimizador Adam y la función de pérdida MSE."""
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse')
    return model

def train_model(model, train_data, train_targets, epochs=10, batch_size=32, validation_split=0.1, verbose=0):
    """Entrena el modelo feedforward."""
    history = model.fit(train_data, train_targets, epochs=epochs, batch_size=batch_size,
                        validation_split=validation_split, verbose=verbose)
    return history

def evaluate_model(model, test_data, test_targets, verbose=0):
    """Evalúa el modelo en el conjunto de prueba."""
    loss = model.evaluate(test_data, test_targets, verbose=verbose)
    return loss

def predict_sequence(model, sequence):
    """Realiza una predicción para una única secuencia."""
    sequence = np.array(sequence, dtype=np.float32).reshape(1, -1)
    prediction = model.predict(sequence)[0][0]
    return prediction