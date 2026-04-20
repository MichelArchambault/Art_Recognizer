from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

MODEL_PATH = 'art_classifier_model.h5'
if os.path.exists(MODEL_PATH):
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
else:
    print("Model not found. Please train the model first.")
    model = None

IMG_SIZE = (128, 128)
CLASS_NAMES = ['Abstract_Expressionism', 'Action_painting', 'Analytical_Cubism', 'Art_Nouveau_Modern', 'Baroque', 'Color_Field_Painting', 'Contemporary_Realism', 'Cubism', 'Early_Renaissance', 'Expressionism', 'Fauvism', 'High_Renaissance', 'Impressionism', 'Mannerism_Late_Renaissance', 'Minimalism', 'Naive_Art_Primitivism', 'New_Realism', 'Northern_Renaissance', 'Pointillism', 'Pop_Art', 'Post_Impressionism', 'Realism', 'Rococo', 'Romanticism', 'Symbolism', 'Synthetic_Cubism', 'Ukiyo_e']  # Update if needed

EXAMPLE_PIECES = [
    {
        'style': 'Impressionism',
        'title': 'Water Lilies',
        'artist': 'Claude Monet',
        'image_url': 'https://upload.wikimedia.org/wikipedia/commons/1/14/Claude_Monet_-_Water_Lilies_-_Google_Art_Project.jpg'
    },
    {
        'style': 'Ukiyo-e',
        'title': 'The Great Wave off Kanagawa',
        'artist': 'Katsushika Hokusai',
        'image_url': 'https://upload.wikimedia.org/wikipedia/commons/0/0a/The_Great_Wave_off_Kanagawa.jpg'
    },
    {
        'style': 'Baroque',
        'title': 'The Calling of Saint Matthew',
        'artist': 'Caravaggio',
        'image_url': 'https://upload.wikimedia.org/wikipedia/commons/6/6d/The_Calling_of_Saint_Matthew.jpg'
    },
    {
        'style': 'Renaissance',
        'title': 'Mona Lisa',
        'artist': 'Leonardo da Vinci',
        'image_url': 'https://upload.wikimedia.org/wikipedia/commons/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg'
    },
    {
        'style': 'Post-Impressionism',
        'title': 'Starry Night Over the Rhone',
        'artist': 'Vincent van Gogh',
        'image_url': 'https://upload.wikimedia.org/wikipedia/commons/9/94/Starry_Night_Over_the_Rhone.jpg'
    },
    {
        'style': 'Expressionism',
        'title': 'The Scream',
        'artist': 'Edvard Munch',
        'image_url': 'https://upload.wikimedia.org/wikipedia/commons/f/f4/The_Scream.jpg'
    }
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def classify_image(image_path):
    if model is None:
        return "Model not loaded", 0.0
    
    img = keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, 0) / 255.0

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    predicted_class = CLASS_NAMES[np.argmax(score)]
    confidence = 100 * np.max(score)

    return predicted_class, confidence

@app.route('/')
def index():
    return render_template('index.html', class_names=CLASS_NAMES, example_pieces=EXAMPLE_PIECES)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        predicted_class, confidence = classify_image(filepath)
        
        with open(filepath, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        return render_template('result.html', 
                             predicted_class=predicted_class, 
                             confidence=confidence, 
                             img_data=img_data)
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True, port=5001)