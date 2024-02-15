import base64
import urllib.parse
from flask import Flask, request, jsonify, send_file
from gradio_client import Client
import os
import shutil
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, storage, db

app = Flask(__name__)

# Initialize Firebase
cred = credentials.Certificate("bening-app-firebase-adminsdk-42cyk-19c0554d22.json")
firebase_admin.initialize_app(cred, {'storageBucket': 'bening-app.appspot.com', 'databaseURL': 'https://bening-app-default-rtdb.asia-southeast1.firebasedatabase.app'})

# Define the output directory in your project
output_directory = "output"

firebase_folder = "resultImages"  # Specify the folder name you want in Firebase Storage

@app.route('/predict', methods=['POST'])
def predict():
    # Parse parameters from JSON payload
    data = request.get_json()

    if data:
        # Parse parameters from query parameters
        image_path = request.args.get('image', data.get('image', None))
        uid_user = request.args.get('uid', data.get('uid', None))
        image_Id = request.args.get('image_id', data.get('image_id', None))
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
            new_filename = f"{current_datetime}.png"
            destination_path = os.path.join(output_directory, new_filename)

            # Move the file from the source location to the output directory with the new name
            shutil.move(result, destination_path)

            # Upload the image to Firebase Storage in the "images" folder
            bucket = storage.bucket()
            blob = bucket.blob(firebase_folder + "/" + new_filename)
            
            try:
                blob.upload_from_filename(destination_path)
            except Exception as e:
                return jsonify({'error': f'Firebase upload failed: {str(e)}'}), 500

            # Get the URL of the uploaded image from Firebase Storage
            bucket_name = blob.bucket.name
            blob_name = blob.name
            safe_blob_name = urllib.parse.quote(blob_name, safe='')
            image_url = f"https://firebasestorage.googleapis.com/v0/b/{bucket_name}/o/{safe_blob_name}?alt=media"
            # Write image_path and image_url to Firebase Realtime Database
            ref = db.reference('/')
            # Get a reference to the 'users/uid/firstImages' node
            user_ref = ref.child(f'users/{uid_user}/resultImages')
        
            # Set the generated filename under the image ID
            user_ref.child(image_Id).set(new_filename)
            
            print(new_filename)

            # Return the JSON with the image URL
            return jsonify({'result_image_path': new_filename})
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

    # Create a JSON response with appropriate Content-Type header
    return jsonify({'encoded_image': encoded_image}), 200, {'Content-Type': 'application/json'}

@app.route('/status', methods=['GET'])
def get_server_status():
    print('Server is up and running')
    return jsonify({'status': 'Server is up and running'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0')
