import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from LSTM import LSTMLayer

def load_data(csv_path, time_steps, input_dim, num_classes):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Assuming the CSV has columns for features and labels
    # Adjust column names and indices as per your CSV structure
    features = df.iloc[:, :input_dim].values
    labels = df.iloc[:, input_dim].values
    
    # Reshape features to match the required dimensions
    num_samples = len(features) // time_steps
    X = features[:num_samples * time_steps].reshape(num_samples, time_steps, input_dim).astype(np.float32)
    
    # Reshape labels to match the required dimensions
    y = labels[:num_samples * time_steps].reshape(num_samples, time_steps).astype(np.float32)
    
    # Ensure labels are integers within the range [0, num_classes)
    y = np.clip(y.astype(int), 0, num_classes - 1)
    
    return X, y

def create_model(input_shape, lstm_units, num_classes):
    model = tf.keras.Sequential([
        LSTMLayer(lstm_units),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, activation='relu')),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_classes, activation='softmax'))
    ])
    
    model.build(input_shape)
    return model

def compile_model(model, learning_rate=0.001):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size):
    history = model.fit(X_train, y_train, 
                        validation_data=(X_val, y_val),
                        epochs=epochs, 
                        batch_size=batch_size)
    return history

def evaluate_model(model, X_test, y_test):
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    return test_loss, test_accuracy

def main(csv_path, time_steps, input_dim, lstm_units, num_classes, epochs, batch_size, learning_rate):
    # Load data
    X, y = load_data(csv_path, time_steps, input_dim, num_classes)

    # Split the data into training, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

    # Create the model
    model = create_model((None, time_steps, input_dim), lstm_units, num_classes)
    model.summary()

    # Compile the model
    compile_model(model, learning_rate)

    # Train the model
    history = train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size)

    # Evaluate the model
    test_loss, test_accuracy = evaluate_model(model, X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")

    return model, history

if __name__ == "__main__":
    # Example usage
    model, history = main(
        csv_path='path/to/your/data.csv',
        time_steps=20,
        input_dim=10,
        lstm_units=64,
        num_classes=5,
        epochs=10,
        batch_size=32,
        learning_rate=0.001
    )