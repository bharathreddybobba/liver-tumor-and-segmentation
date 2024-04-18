from flask import Flask, render_template, request, jsonify, redirect, url_for
import numpy as np
from PIL import Image
import io
import base64
import cv2
import tensorflow as tf

app = Flask(__name__)


model_paths = {
    'liver_tumor': 'liver_tumor.h6',
    'liver_segmentation': 'liver_segmentation.h7',
    'couinaud_segmentation': 'couinaud_segmentation.h5'
}

models = {}
for model_name, model_path in model_paths.items():
    models[model_name] = tf.keras.models.load_model(model_path)

# Define a route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Define a function to preprocess the image
def preprocess_image(file):
    img = Image.open(io.BytesIO(file.read()))
    img = img.resize((128, 128))  # Resize image to match model input size
    img = np.array(img) / 255.0  # Normalize pixel values
    return img

# Define a function to encode image to base64
def img_to_base64(img):
    img_str = base64.b64encode(cv2.imencode('.jpg', img)[1]).decode('utf-8')
    return img_str

# Define a route for model prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the file from the POST request
        file = request.files['file']
        # Get the selected model from the POST request
        selected_model = request.form['selected_model']
        # Check if file is uploaded
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400
        # Preprocess the image
        img = preprocess_image(file)
        # Make prediction using the selected model
        model = models[selected_model]
        prediction = model.predict(np.array([img]))[0]  # Assuming single prediction
        # Determine the result
        result = "Affected by liver tumor" if prediction > 0.5 else "Not affected by liver tumor"
        # Encode the image to base64
        img_str = img_to_base64(img)
        # Redirect to the result page with image data and result
        return redirect(url_for('result', image_data=img_str, result=result))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Define a route to render the result page
@app.route('/result')
def result():
    # Get image data and result from URL parameters
    image_data = request.args.get('image_data')
    result = request.args.get('result')
    if not image_data:
        return jsonify({'error': 'No image data provided'}), 400
    return render_template('result.html', image_data=image_data, result=result)

if __name__ == "__main__":
    app.run(debug=True)
