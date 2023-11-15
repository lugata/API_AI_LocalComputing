import requests

# Define the image URL
image_path = 'https://firebasestorage.googleapis.com/v0/b/unblurapp-4ed35.appspot.com/o/images%2Ftest.jpg?alt=media&token=219faf08-1c97-4ed2-9a67-43e26f46568a&_gl=1*stjmtm*_ga*MzE4NTIxMzA4LjE2ODEyMTgzNTg.*_ga_CW55HF8NVT*MTY5ODQyNDg3OC4zNi4x.LjE2OTg0MjQ4ODUuNTMuMC4w.jpg'

# Define the API endpoint URL
api_url = 'http://192.168.2.57:8080/predict'

# Define the parameters you want to send
params = {
    'image': image_path,
    'background_enhance': True,  # Set to True or False
    'face_upsample': True,  # Set to True or False
    'rescaling_factor': 2,  # Specify the rescaling factor
    'codeformer_fidelity': 0.6  # Specify the CodeFormer fidelity
}

# Send the POST request
response = requests.post(api_url, json=params)

# Check the response
if response.status_code == 200:
    # Request was successful
    data = response.json()
    image_path = data.get('image_path')
    print(f'Processed image URL: {image_path}')
else:
    # Request failed
    print(f'Error: {response.text}')
