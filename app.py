from flask import Flask, request, jsonify
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, BatchNormalization, GlobalAveragePooling2D, Dense, Add, Concatenate
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.models import Model
from keras.layers import Add
from keras.layers import GlobalAveragePooling2D
from keras.layers import Conv2D, Add, MaxPooling2D, Activation, Dense
from keras.layers import Input, BatchNormalization, GlobalAveragePooling2D,Concatenate
from keras.models import Model
from flask_cors import CORS
import numpy as np
from PIL import Image
import os
import json
from scipy import io
import matplotlib.pyplot as plt
import multiprocessing
# import warnings
# warnings.filterwarnings("ignore")


app = Flask(__name__)
CORS(app)


# Define constants
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
NUM_CLASSES = 2


# Function to define the neural network model architecture
def EEG_MODEL(input_shape, num_classes):
    input_1 = Input(shape=input_shape, name='input_1')
    conv1 = Conv2D(64,(3,3), activation = LeakyReLU(alpha=0.02))(input_1)
    x = BatchNormalization()(conv1)
    
    x1 = Conv2D(64, (3, 3), dilation_rate = 1, padding = 'same')(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation(LeakyReLU(alpha=0.02))(x1)
    x1 = Add()([x, x1])
    
    x2 = Conv2D(64, (3, 3), dilation_rate = 1, padding = 'same')(x1)
    x2 = BatchNormalization()(x2)
    x2 = Activation(LeakyReLU(alpha=0.02))(x2)
    x2 = Add()([x, x2])
    
    x11 = Conv2D(64, (3, 3), dilation_rate = 2, padding = 'same')(x)
    x11 = BatchNormalization()(x11)
    x11 = Activation(LeakyReLU(alpha=0.02))(x11)
    x11 = Add()([x, x11])
    
    x21 = Conv2D(64, (3, 3), dilation_rate = 2, padding = 'same')(x11)
    x21 = BatchNormalization()(x21)
    x21 = Activation(LeakyReLU(alpha=0.02))(x21)
    x21 = Add()([x, x21])
    
    x111 = Conv2D(64, (3, 3), dilation_rate = 3, padding = 'same')(x)
    x111 = BatchNormalization()(x111)
    x111 = Activation(LeakyReLU(alpha=0.02))(x111)
    x111 = Add()([x, x111])
    
    x211 = Conv2D(64, (3, 3), dilation_rate = 3, padding = 'same')(x111)
    x211 = BatchNormalization()(x211)
    x211 = Activation(LeakyReLU(alpha=0.02))(x211)
    x211 = Add()([x, x211])
    
    con = Concatenate()([x2, x21, x211])
    x = MaxPooling2D(pool_size = (2,2), strides = 2)(con)
    
    conv1 = Conv2D(128,(3,3), activation = LeakyReLU(alpha=0.02))(x)
    x = BatchNormalization()(conv1)
    
    x1 = Conv2D(128, (3, 3), dilation_rate = 1, padding = 'same')(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation(LeakyReLU(alpha=0.02))(x1)
    x1 = Add()([x, x1])
    
    x2 = Conv2D(128, (3, 3), dilation_rate = 1, padding = 'same')(x1)
    x2 = BatchNormalization()(x2)
    x2 = Activation(LeakyReLU(alpha=0.02))(x2)
    x2 = Add()([x, x2])
    
    x11 = Conv2D(128, (3, 3), dilation_rate = 2, padding = 'same')(x)
    x11 = BatchNormalization()(x11)
    x11 = Activation(LeakyReLU(alpha=0.02))(x11)
    x11 = Add()([x, x11])
    
    x21 = Conv2D(128, (3, 3), dilation_rate = 2, padding = 'same')(x11)
    x21 = BatchNormalization()(x21)
    x21 = Activation(LeakyReLU(alpha=0.02))(x21)
    x21 = Add()([x, x21])
    
    x111 = Conv2D(128, (3, 3), dilation_rate = 3, padding = 'same')(x)
    x111 = BatchNormalization()(x111)
    x111 = Activation(LeakyReLU(alpha=0.02))(x111)
    x111 = Add()([x, x111])
    
    x211 = Conv2D(128, (3, 3), dilation_rate = 3, padding = 'same')(x111)
    x211 = BatchNormalization()(x211)
    x211 = Activation(LeakyReLU(alpha=0.02))(x211)
    x211 = Add()([x, x211])
    
    con = Concatenate()([x2, x21, x211])
    x = MaxPooling2D(pool_size = (2,2), strides = 2)(con)
    
    conv1 = Conv2D(128,(3,3), activation = LeakyReLU(alpha=0.02))(x)
    x = BatchNormalization()(conv1)
    
    x1 = Conv2D(128, (3, 3), dilation_rate = 1, padding = 'same')(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation(LeakyReLU(alpha=0.02))(x1)
    x1 = Add()([x, x1])
    
    x2 = Conv2D(128, (3, 3), dilation_rate = 1, padding = 'same')(x1)
    x2 = BatchNormalization()(x2)
    x2 = Activation(LeakyReLU(alpha=0.02))(x2)
    x2 = Add()([x, x2])
    
    x11 = Conv2D(128, (3, 3), dilation_rate = 2, padding = 'same')(x)
    x11 = BatchNormalization()(x11)
    x11 = Activation(LeakyReLU(alpha=0.02))(x11)
    x11 = Add()([x, x11])
    
    x21 = Conv2D(128, (3, 3), dilation_rate = 2, padding = 'same')(x11)
    x21 = BatchNormalization()(x21)
    x21 = Activation(LeakyReLU(alpha=0.02))(x21)
    x21 = Add()([x, x21])
    
    x111 = Conv2D(128, (3, 3), dilation_rate = 3, padding = 'same')(x)
    x111 = BatchNormalization()(x111)
    x111 = Activation(LeakyReLU(alpha=0.02))(x111)
    x111 = Add()([x, x111])
    
    x211 = Conv2D(128, (3, 3), dilation_rate = 3, padding = 'same')(x111)
    x211 = BatchNormalization()(x211)
    x211 = Activation(LeakyReLU(alpha=0.02))(x211)
    x211 = Add()([x, x211])
    
    con = Concatenate()([x2, x21, x211])
    g3 = GlobalAveragePooling2D()(con)
    d3 = Dense(512, activation = 'relu')(g3)
    d4 = Dense(256, activation = 'relu')(g3)
    d5 = Dense(2, activation = 'softmax')(d4)

    model = Model(inputs= input_1, outputs= d5)
    return model

# Load your trained model
input_shape = (128, 128, 3)
model = EEG_MODEL(input_shape, NUM_CLASSES)
model.load_weights("./best_weights.h5")

import tempfile

def preprocess_and_convert_to_arrays(matlab_file):
    """
    Load EEG data from a MATLAB file and preprocess it.

    Parameters:
    matlab_file (FileStorage): Uploaded MATLAB file.

    Returns:
    numpy.ndarray: Preprocessed EEG data.
    """
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        file_path = temp_file.name
        matlab_file.save(file_path)
        mat_data = io.loadmat(file_path)
    # Close the file handle
    temp_file.close()
    # Remove the temporary file
    os.remove(file_path)
    eeg_data = mat_data['EEG']['data'][0, 0]
    return eeg_data



def convert_to_desired_format(data):
    """
    Convert the given data into the desired format.

    Parameters:
    data (list): A list of numpy arrays containing EEG data.

    Returns:
    numpy.ndarray: An array containing the modified data.
    """
    modified_data = []
    for i in range(len(data)):
        modified_data.append(data[i][:32, :])
    return np.array(modified_data)

def preprocess_data(data):
    """
    Preprocess the given data.

    Parameters:
    data (list): A list of numpy arrays containing EEG data.

    Returns:
    list: Preprocessed EEG data.
    """
    preprocessed_data = []
    for i in range(len(data)):
        std = np.std(data[i], axis=1)
        mean = np.mean(data[i], axis=1)
        data[i] = (data[i].transpose() - mean.transpose()).transpose()
        data[i] = (data[i].transpose() / std.transpose()).transpose()
        preprocessed_data.append(data[i])
    return preprocessed_data

def process_j(j, preprocessed_data, winSize, stride):
    try:
        count = 0
        WData_j = None
        for i in range(0, np.shape(preprocessed_data[j])[1] - winSize, stride):
            count += 1
            if WData_j is None:
                WData_j = np.reshape(preprocessed_data[j][:, i:i + winSize], (32, np.shape(preprocessed_data[j][:, i:i + winSize])[1], 1))
            else:
                WData_j = np.dstack((WData_j, preprocessed_data[j][:, i:i + winSize]))
        return count, WData_j
    except Exception as e:
        print(f"Exception occurred in process_j: {e}")
        return 0, None


def process_data(file_path):
    """
    Process the data in parallel.

    Parameters:
    file_path (str): Path to the MATLAB file.

    Returns:
    numpy.ndarray: Processed EEG data.
    """
    try:
        eeg_data = preprocess_and_convert_to_arrays(file_path)
        modified_data_array = convert_to_desired_format([eeg_data])
        preprocessed_data = preprocess_data(modified_data_array)

        winSize = 512 * 4
        stride = 512 * 1
        
        # Process data in parallel
        pool = multiprocessing.Pool()
        results = pool.starmap(process_j, [(j, preprocessed_data, winSize, stride) for j in range(len(preprocessed_data))])
        pool.close()
        pool.join()

        processed_data = None
        for count, WData_j in results:
            if processed_data is None:
                processed_data = WData_j
            else:
                processed_data = np.dstack((processed_data, WData_j))
        
        return np.transpose(processed_data, (2, 0, 1))
    except Exception as e:
        print(f"Exception occurred in process_data: {e}")
        return None

from io import BytesIO
import six
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import random

def create_images(data):
    """
    Create an image using pcolormesh.

    Parameters:
    data (numpy.ndarray): Data array.

    Returns:
    PIL.Image.Image: PIL Image object.
    """
    # Select a random slice
    random_slice = random.randint(0, data.shape[0] - 1)
    data_slice = data[random_slice, :, :]

    # Create a color mesh plot using 'viridis' colormap
    plt.figure(figsize=(data_slice.shape[1] / 100, data_slice.shape[0] / 100), dpi=100)  # Adjust figsize and dpi as needed
    plt.pcolormesh(data_slice, cmap='viridis')
    plt.axis('off')
    
    # Convert the plot to a PIL Image
    buf = six.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)  # Adjust dpi and bbox_inches as needed
    buf.seek(0)
    img = Image.open(buf)
    
    # Close the plot to release memory
    plt.close()
    
    return img




def process_images(images):
    """
    Resize and convert images to the desired format.

    Parameters:
    images (List[PIL.Image]): List of input images.

    Returns:
    List[numpy.ndarray]: List of resized and converted images.
    """
    processed_images = []
    for img in images:
        img_resized = img.resize((128, 128))
        if img_resized.mode != 'RGB':
            img_resized = img_resized.convert('RGB')
        img_array = np.array(img_resized)

        # Normalize pixel values to [0, 1]
        img_array = img_array.astype('float32') / 255.0
        
        # img_array = np.expand_dims(img_array, axis=0)

        processed_images.append(img_array)
    return processed_images

# Function to make predictions using the model
def predict_eeg_data(eeg_data):
    print("Received EEG data shape:", eeg_data.shape)  # Debug karenge ab
    eeg_data_reshaped = eeg_data.reshape(-1, 128, 128, 3)
    prediction = model.predict(eeg_data_reshaped)
    # prediction = model.predict(eeg_data) 
    print("Prediction shape:", prediction.shape)
    return prediction

# Route for handling prediction
@app.route('/predict', methods=['POST'])
def predict_disease():
    try:
        name = request.form['name']
        # Extract form data
        age = request.form['age']
        history = request.form['history']
        matlab_file = request.files['matlab_file']
        print(matlab_file)
        # Check if the file is uploaded
        if 'matlab_file' not in request.files:
            return "No MATLAB file uploaded"
        
        # Check if the file is valid
        if matlab_file.filename == '':
            return "No MATLAB file selected"
        
        # Process the MATLAB file
        processed_data = process_data(matlab_file)
        images = [create_images(processed_data)]  
        processed_images = process_images(images)
        predictions = predict_eeg_data(processed_images[0])  # Process only the first image
        results = []
            
        # Process each prediction
        for prediction in predictions:
            print("Prediction:", prediction)
            class_index = np.argmax(prediction)
            print("Class Index:", class_index)
            result = "Parkinson's Disease Detected" if class_index == 1 else "Healthy"
            results.append(result)
        
        # Display the results
        # print("Results:", results)
        # print("Sum of probabilities for Healthy class:", sum_healthy)
        # print("Sum of probabilities for Disease class:", sum_disease)
        # print(predictions)
    
        
        # Create a JSON object with form data and prediction result
        prediction_data = {
            'name': name,
            'age': age,
            'history': history,
            'result': result
        }
        
        # Save prediction data to a JSON file
        save_to_json(prediction_data)
        print(result)
        # Return prediction result
        return jsonify(result)
    except Exception as e:
        return jsonify("Prediction failed")
# Function to save prediction data to a JSON file
def save_to_json(data):
    filename = 'predictions.json'
    with open(filename, 'a') as file:
        json.dump(data, file)
        file.write('\n')

if __name__ == '__main__':
    app.run(debug=True,port=8080)

