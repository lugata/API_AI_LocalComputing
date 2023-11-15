from flask import Flask, request, jsonify, send_file
from gradio_client import Client
import os
import shutil
from datetime import datetime

app = Flask(__name__)

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
            getPredict = f"http://{IpAddress}:{Port}/predict/{new_filename}"
            # Move the file from the source location to the output directory with the new name
            shutil.move(result, destination_path)

            return jsonify({'result_image_path': getPredict})
        else:
            return jsonify({'error': 'Missing parameters'})
    else:
        return jsonify({'error': 'Missing JSON payload or query parameters'})


@app.route('/predict/<filename>', methods=['GET'])
def get_generated_image(filename):
    # Construct the path to the image based on the provided filename
    image_path = os.path.join(output_directory, filename)
    
    # Check if the file exists
    if os.path.exists(image_path):
        # Send the file as a response
        return send_file(image_path, mimetype='image/png')
    else:
        return jsonify({'error': 'Image not found'})

if __name__ == '__main__':
    app.run(host=IpAddress, port=Port)
