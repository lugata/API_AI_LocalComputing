import base64
from flask import Flask, request, jsonify, send_file
from gradio_client import Client
import os
import shutil
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, storage

app = Flask(__name__)

# Initialize Firebase
cred = credentials.Certificate("bening-app-firebase-adminsdk-42cyk-19c0554d22.json")
firebase_admin.initialize_app(cred, {'storageBucket': 'bening-app.appspot.com'})

IpAddress = "192.168.2.51"
Port = 8080
# Define the output directory in your project
output_directory = "output"


@app.route('/predict', methods=['POST'])
def predict():
    # Parse parameters from JSON payload
    data = request.get_json()

    if data:
        # Parse parameters from query parameters
        image_path = request.args.get('image', data.get('image', None))
        background_enhance = request.args.get('background_enhance', data.get('background_enhance', None))
        face_upsample = request.args.get('face_upsample', data.get('face_upsample', None))
        rescaling_factor = request.args.get('rescaling_factor', data.get('rescaling_factor', None))
        codeformer_fidelity = request.args.get('codeformer_fidelity', data.get('codeformer_fidelity', None))

        if image_path and background_enhance and face_upsample and rescaling_factor and codeformer_fidelity:
            # Continue with the rest of your code

            client = Client("http://127.0.0.1:7860/")
            result = client.predict(
                image_path,
                background_enhance,
                face_upsample,
                rescaling_factor,
                codeformer_fidelity,
                api_name="/predict"
            )

            # Ensure the output directory exists
            os.makedirs(output_directory, exist_ok=True)

            # Generate a new filename based on the current date and time
            current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
            new_filename = f"UnBlurApp_{current_datetime}.png"
            destination_path = os.path.join(output_directory, new_filename)

            # Move the file from the source location to the output directory with the new name
            shutil.move(result, destination_path)

            # Upload the image to Firebase Storage
            bucket = storage.bucket()
            blob = bucket.blob(new_filename)
            blob.upload_from_filename(destination_path)

            # Get the URL of the uploaded image
            image_url = blob.public_url

            # Return the JSON with the image URL
            return jsonify({'result_image_path': image_url})
        else:
            return jsonify({'error': 'Missing parameters'})
    else:
        return jsonify({'error': 'Missing JSON payload or query parameters'})


@app.route('/predict/<filename>', methods=['GET'])
def get_generated_image(filename):
    # Construct the path to the image based on the provided filename
    image_path = os.path.join(output_directory, filename)

    # Check if the file exists
    if not os.path.exists(image_path):
        return jsonify({'error': 'Image not found'}), 404

    # Read the image file
    with open(image_path, 'rb') as image_file:
        # Encode the image in base64
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        print(encoded_image)
    # Create a JSON response
    return jsonify(encoded_image)

if __name__ == '__main__':
    app.run(host=IpAddress, port=Port)
