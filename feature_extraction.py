from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense

def create_cnn_model(input_shape):
    model = Sequential([
        Conv2D(16, (5, 5), activation='relu', input_shape=input_shape),
        AveragePooling2D(pool_size=(2, 2)),
        Conv2D(32, (5, 5), activation='relu'),
        AveragePooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')  # For binary classification (fracture or not)
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Example usage
model = create_cnn_model((128, 128, 1))  # assuming grayscale images with size 128x128
