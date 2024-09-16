import time
# Start measuring time
start_time = time.time()


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
from PIL import Image
from tqdm import tqdm

# Constants
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 10
SIZE_OF_TRAINING_DATA = 2000


# Load train.csv
train_df = pd.read_csv('dataset/train.csv').head(SIZE_OF_TRAINING_DATA)

# Helper function to download and preprocess images
def get_image(url):
    try:
        img = Image.open(os.path.join('images/', f"{idx}.jpg"))
        img = img.resize(IMG_SIZE)
        img = img_to_array(img) / 255.0  # Normalize the image
        if img.shape == (*IMG_SIZE, 3):
            return img
        else:
            print(f"Unexpected image shape for index {idx}: {img.shape}")
            return np.zeros((*IMG_SIZE, 3))  # Return black image if shape is incorrect
    except Exception as e:
        print(f"Error In Loading {idx}: {e}")
        return np.zeros((*IMG_SIZE, 3))  # Return a black image on error


# Load images and corresponding labels (entity_value)
images = []
labels = []
for idx, row in tqdm(train_df.iterrows()):
    image_url = row['image_link']
    entity_value = row['entity_value']  # Example: "34 gram"
    # Download and preprocess image
    img = get_image(image_url)
    images.append(img)
    # Extract and process the numerical part of the entity_value
    try:
        value = float(entity_value.split()[0])
        labels.append(value)
    except:
        labels.append(0.0)  # If parsing fails, assign 0.0 as default

# Convert to numpy arrays
x = np.array(images)
y = np.array(labels)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# Build a simple CNN model from scratch
def create_model():
    model = Sequential()
    # Convolutional layers
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(*IMG_SIZE, 3)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # Flatten the feature maps and add Dense layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Dropout to prevent overfitting

    # Output layer for regression (predicting the float value)
    model.add(Dense(1))  # No activation function for regression

    return model

# Compile the model
model = create_model()
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model-
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val))

# Evaluate the model on validation data
val_loss, val_mae = model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss}, Validation MAE: {val_mae}")

# Save the trained model
model.save(f'model_train_on_{SIZE_OF_TRAINING_DATA}.h5')

# End measuring time
end_time = time.time()

# Calculate the time taken
time_taken = end_time - start_time

print(f"Time taken to train on {SIZE_OF_TRAINING_DATA} Images: {time_taken} seconds")
