import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from data_util import generate_synthetic_sequence_data, split_data
from model import create_ff_model, compile_model, train_model, evaluate_model, predict_sequence

# --- Configuración de la página ---
st.set_page_config(page_title="Predicción de Secuencias Cortas", layout="centered")
st.title("🔮 Predicción de Secuencias Numéricas Cortas")
st.markdown("Un modelo feedforward simple para predecir el siguiente valor (en este caso, la suma) de una secuencia numérica.")

# --- Parámetros configurables ---
sequence_length = st.sidebar.slider("Longitud de la Secuencia:", min_value=2, max_value=5, value=3)
num_sequences = st.sidebar.slider("Número de Secuencias:", min_value=100, max_value=2000, value=1000, step=100)
epochs = st.sidebar.slider("Número de Épocas:", min_value=10, max_value=100, value=50, step=10)
learning_rate = st.sidebar.slider("Tasa de Aprendizaje:", min_value=0.0001, max_value=0.01, value=0.001, step=0.0001)

# --- Generación y división de datos ---
with st.spinner("Generando y dividiendo datos..."):
    data, targets = generate_synthetic_sequence_data(num_sequences, sequence_length)
    train_data, train_targets, test_data, test_targets = split_data(data, targets)
st.success("✅ Datos generados y divididos.")

# --- Estado de la sesión para el modelo ---
if 'model' not in st.session_state:
    st.session_state.model = None

if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# --- Entrenamiento del modelo ---
if st.sidebar.button("🚀 Entrenar Modelo"):
    with st.spinner(f"Entrenando el modelo durante {epochs} épocas..."):
        model = create_ff_model(sequence_length)
        model = compile_model(model, learning_rate=learning_rate)
        history = train_model(model, train_data, train_targets, epochs=epochs, verbose=1)
        st.session_state.model = model
        st.session_state.model_trained = True
        st.success("✅ ¡Modelo entrenado!")

        # Mostrar la pérdida durante el entrenamiento
        fig, ax = plt.subplots()
        ax.plot(history.history['loss'], label='Pérdida (Entrenamiento)')
        ax.plot(history.history['val_loss'], label='Pérdida (Validación)')
        ax.set_xlabel('Época')
        ax.set_ylabel('Error Cuadrático Medio (MSE)')
        ax.legend()
        st.pyplot(fig)

# --- Predicción ---
st.header("🔮 Predicción de Nueva Secuencia")
st.markdown("Introduce una secuencia numérica (separada por comas) para predecir el siguiente valor (su suma).")
new_sequence_str = st.text_input(f"Secuencia de {sequence_length} números:", "")

if st.session_state.model_trained and st.session_state.model is not None:
    if st.button("✨ Predecir"):
        try:
            new_sequence = [float(x.strip()) for x in new_sequence_str.split(',')]
            if len(new_sequence) == sequence_length:
                prediction = predict_sequence(st.session_state.model, new_sequence)
                st.subheader(f"Predicción para la secuencia: {new_sequence}")
                st.success(f"El modelo predice: **{prediction:.2f}**")
            else:
                st.error(f"Por favor, introduce una secuencia de exactamente {sequence_length} números.")
        except ValueError:
            st.error("Por favor, introduce números válidos separados por comas.")
else:
    st.info("Por favor, entrena el modelo primero en la barra lateral.")