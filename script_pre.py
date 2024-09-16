import time
# Start measuring time
start_time = time.time()

import os
import random
import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

SIZE_OF_PREDICTION_DATA = 100
IMG_SIZE = (256, 256)  # Must match the input size of your trained model
MODEL_PATH = 'model_train_on_2000.h5'  # Path to the saved model
IMAGE_DIR = 'test_cases/'  # Directory where images are saved
UNITS = ["inch", "cm", "gram", "kg", "volt", "watt"]
ALLOWED_UNITS = {
    'width': ['centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'],
    'depth': ['centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'],
    'height': ['centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'],
    'item_weight': ['gram',
        'kilogram',
        'microgram',
        'milligram',
        'ounce',
        'pound',
        'ton'],
    'maximum_weight_recommendation': ['gram',
        'kilogram',
        'microgram',
        'milligram',
        'ounce',
        'pound',
        'ton'],
    'voltage': ['kilovolt', 'millivolt', 'volt'],
    'wattage': ['kilowatt', 'watt'],
    'item_volume': ['centilitre',
        'cubic foot',
        'cubic inch',
        'cup',
        'decilitre',
        'fluid ounce',
        'gallon',
        'imperial gallon',
        'litre',
        'microlitre',
        'millilitre',
        'pint',
        'quart']
}


# Load the trained model
def preprocess_image(image_path):
    """
    Preprocess the image to feed into the model.
    """
    try:
        img = Image.open(image_path)
        img = img.resize(IMG_SIZE)
        img = img_to_array(img) / 255.0  # Normalize image to [0, 1] range
        return np.expand_dims(img, axis=0)  # Add batch dimension
    except Exception as e:
        print(f"Error loading image {image_path}")
        black_img = np.zeros((*IMG_SIZE, 3))  # A black image with the same size and 3 color channels
        return np.expand_dims(black_img, axis=0)  # Add batch dimension

def get_image(idx):
    try:
        return os.path.join(IMAGE_DIR, f"{idx}.jpg")
    except Exception as e:
        return ""

def predictor(entity_name, idx):
    '''
    Call the model and predict the entity value.
    '''
    # Get the image path (you should have already downloaded the images to IMAGE_DIR)
    image_path = get_image(idx)
    if image_path is None:
        return ""
    
    # Preprocess the image
    image = preprocess_image(image_path)

    # Predict using the trained model
    prediction_value = model.predict(image)
    return f"{prediction_value[0][0]:.2f}"  # Format the prediction as 'x unit'


def safe_predictor(row):
    try:
        # Run the predictor function and return the result
        res = predictor(row['entity_name'], row.name)
        return res
    except Exception as e:
        print(f"Error processing row {row.name}")
        return ""  # Return empty string on error

model = load_model(MODEL_PATH)
test = pd.read_csv(os.path.join('dataset/', 'test.csv')).head(SIZE_OF_PREDICTION_DATA)
with open('test_out.csv', 'w') as f:
    f.write('index,prediction\n')
    for idx, row in test.iterrows():
        try:
            res = safe_predictor(row)
            f.write(f"{row['index']},{res}\n")
        except Exception as e:
            f.write(f"{row['index']},\n")
            print(f"Error In Processing Image")

# End measuring time
end_time = time.time()

# Calculate the time taken
time_taken = end_time - start_time

print(f"Time taken to predict on {SIZE_OF_PREDICTION_DATA} Images: {time_taken} seconds")
