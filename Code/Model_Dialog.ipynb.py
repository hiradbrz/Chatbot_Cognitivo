# Databricks notebook source
#%pip install torch
#%pip install mlflow transformers
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoTokenizer, AutoModelForCausalLM
import mlflow
from mlflow.pyfunc import PythonModel, PythonModelContext

class DialogModel:
    def __init__(self):
        # Initialize the tokenizer and model from the 'microsoft/DialoGPT-medium' pre-trained model
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
        self.model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-medium')
        
        # Check if CUDA (GPU support) is available and set the device accordingly
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Initialize a variable to store the history of the conversation
        self.chat_history_ids = None

    def generate_response_dia(self, user_input):
        # Tokenize the user's input and append the end of string token
        new_input_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors='pt')
        new_input_ids = new_input_ids.to(self.device)

        # Concatenate the new input with the chat history (if there is an existing history)
        bot_input_ids = torch.cat([self.chat_history_ids, new_input_ids], dim=-1) if self.chat_history_ids is not None else new_input_ids

        # Generate a response using the model
        with torch.no_grad():
            bot_outputs = self.model.generate(bot_input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)

        # Update the chat history with the generated response
        self.chat_history_ids = bot_outputs if self.chat_history_ids is None else bot_outputs[:, self.chat_history_ids.shape[-1]:]

        # Decode the model's output to a human-readable format
        response = self.tokenizer.decode(bot_outputs[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        return response


class DialogModelWrapper(DialogModel, PythonModel):
    def predict(self, context: PythonModelContext, model_input):
        # Extract the text input from the model_input dictionary
        text_input = model_input.get('text')
        if not isinstance(text_input, str):
            raise ValueError("Input must be a string.")

        # Generate response using the extracted text
        return super().generate_response_dia(text_input)


# Log the model in MLflow
with mlflow.start_run():
    dialog_model = DialogModelWrapper()
    mlflow.pyfunc.log_model("dialog_model", python_model=dialog_model)

# Register the model in MLflow Model Registry
# Replace 'your_run_id_here' with the actual run ID from MLflow
run_id = "47042f796b474229a5a2edc9e1aa125b"
mlflow.register_model(f"runs:/{run_id}/dialog_model", "DialogModel_Registry")


# COMMAND ----------

import requests
import torch

from transformers import AutoTokenizer

# The URL for the model endpoint you've created
endpoint_url = 'https://adb-1012386050250820.0.azuredatabricks.net/serving-endpoints/Dialog_Model/invocations'

# User input as a string
user_input = "Hi"  # Replace this with the actual user input as a string

# Generate a response using the DialogModel
#response = dialog_model.generate_response_dia(user_input)

# Convert the response to a dictionary with the expected format
data = {
    "inputs": generate_response_dia(user_input)
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


# COMMAND ----------

import requests
import json

# Endpoint URL of your model
endpoint_url = "https://adb-1012386050250820.0.azuredatabricks.net/serving-endpoints/Dialog_Model/invocations"

# Prepare the input payload with 'inputs' key
input_text = "Hi"
payload = json.dumps({"inputs": {"text": input_text}})

# Set appropriate headers, including the API key for authentication
api_key = "dapi9c979f949e92eccc6b23128ed50b0b51"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

try:
    # Send the POST request
    response = requests.post(endpoint_url, data=payload, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse and print the response
        response_data = response.json()
        print("Model response:", response_data)
    else:
        print("Failed to get response. Status code:", response.status_code)
        print("Response content:", response.text)  # This will print the response content which might contain error messages
except requests.exceptions.RequestException as e:
    # Catch any request-related errors
    print("Request failed:", e)


# COMMAND ----------

dbutils.library.restartPython()
