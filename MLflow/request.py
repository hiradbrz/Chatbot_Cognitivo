import requests
import torch

from transformers import AutoTokenizer
from your_module import DialogModel  # Import your DialogModel class from your module

# Initialize the DialogModel
dialog_model = DialogModel()

# The URL for the model endpoint you've created
endpoint_url = 'https://adb-1012386050250820.0.azuredatabricks.net/serving-endpoints/Dialog_Model/invocations'

# User input as a string
user_input = "Hi"  # Replace this with the actual user input as a string

# Generate a response using the DialogModel
response = dialog_model.generate_response_dia(user_input)

# Convert the response to a dictionary with the expected format
data = {
    "inputs": [response]
}

# Make a POST request to the endpoint
response = requests.post(
    url=endpoint_url,
    json=data,
    headers={
        'Authorization': 'Bearer dapi9c979f949e92eccc6b23128ed50b0b51',
        'Content-Type': 'application/json'
    }
)

# Check the response
if response.status_code == 200:
    print("Prediction successful:", response.json())
else:
    print("Failed to fetch prediction:", response.text)
