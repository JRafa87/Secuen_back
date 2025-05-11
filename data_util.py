import numpy as np

def generate_synthetic_sequence_data(num_sequences=3000, sequence_length=3):
    """Genera datos sintéticos de secuencias numéricas para predicción."""
    data = []
    targets = []
    for _ in range(num_sequences):
        sequence = np.random.randint(1, 10, size=sequence_length).tolist()
        target = sum(sequence)  # Ejemplo simple: predecir la suma
        data.append(sequence)
        targets.append(target)
    return np.array(data, dtype=np.float32), np.array(targets, dtype=np.float32)

def split_data(data, targets, train_ratio=0.8):
    """Divide los datos en conjuntos de entrenamiento y prueba."""
    split_index = int(len(data) * train_ratio)
    train_data = data[:split_index]
    train_targets = targets[:split_index]
    test_data = data[split_index:]
    test_targets = targets[split_index:]
    return train_data, train_targets, test_data, test_targets