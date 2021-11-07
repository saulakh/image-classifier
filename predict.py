import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import PIL
from PIL import Image
import json
import ntpath
import argparse

IMAGE_SIZE = 224

def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (IMAGE_SIZE,IMAGE_SIZE))
    image /= 255
    image = image.numpy()
    
    return image

def predict(image_path, model, K, class_names):
    image = PIL.Image.open(image_path)
    image_exp = np.expand_dims(np.asarray(image), axis=0)
    processed_image = process_image(image_exp)
    
    # Return top 5 probabilities and class labels
    softmax = model.predict(processed_image) 
    values, idx = tf.nn.top_k(softmax, K)
    
    # Convert from tensor to numpy
    probs = values.numpy()
    labels = idx.numpy()
    
    # Get flower from file name
    title = ntpath.basename(image_path)
    
    # Print top K labels and probabilities
    print("\n", "Image:", title, "\n\n", "Predicted Labels:")
    for i in range(len(labels[0])):
        current_label = labels[0][i] + 1
        current_prob = probs[0][i]
        label = class_names[str(current_label)]
        prob = round(current_prob,2)
        
        print(i+1, ":", label, "-- Probability:", prob)
    
    return

parser = argparse.ArgumentParser()
parser.add_argument('image_path')
parser.add_argument('model_path')
parser.add_argument('--top_k', type=int, default=3)
parser.add_argument('--category_names', default='label_map.json')
args = parser.parse_args()

# Load the Keras model
model = tf.keras.models.load_model(args.model_path,custom_objects={'KerasLayer':hub.KerasLayer})

# Load class names
with open(args.category_names, 'r') as f:
    class_names = json.load(f)

# Predict top K class names
predict(args.image_path, model, args.top_k, class_names)