# Databricks notebook source
#%pip install torch mlflow transformers
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoTokenizer, AutoModelForCausalLM
import mlflow
from mlflow.pyfunc import PythonModel, PythonModelContext

class DialogModel:
    def __init__(self):
        # Initialize the tokenizer and model from the 'microsoft/DialoGPT-medium' pre-trained model
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium',padding_side='left')
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

        # Ensure both tensors are compatible before concatenation
        if self.chat_history_ids is not None and self.chat_history_ids.shape[-1] > 0:
            bot_input_ids = torch.cat([self.chat_history_ids, new_input_ids], dim=-1)
        else:
            bot_input_ids = new_input_ids

        # Generate a response using the model
        with torch.no_grad():
            bot_outputs = self.model.generate(bot_input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)

        # Update the chat history
        self.chat_history_ids = bot_outputs

        # Decode the model's output to a human-readable format
        response = self.tokenizer.decode(bot_outputs[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        return response


# COMMAND ----------

import logging
from mlflow.pyfunc import PythonModel, PythonModelContext

# Ensure basic logging is configured
logging.basicConfig(level=logging.INFO)

class DialogModelWrapper(DialogModel, PythonModel):
    def predict(self, context: PythonModelContext, model_input):
        # Log the incoming model_input
        logging.info(f"Received model input: {model_input}")

        # Extract the text input from the model_input dictionary
        text_input = model_input.get('text')
        if not isinstance(text_input, str):
            logging.error("Input is not a string.")
            raise ValueError("Input must be a string.")

        # Generate response using the extracted text
        response = super().generate_response_dia(text_input)
        logging.info(f"Generated response: {response}")
        return response


# COMMAND ----------

# Log the model in MLflow
with mlflow.start_run():
    dialog_model = DialogModelWrapper()
    mlflow.pyfunc.log_model("dialog_model", python_model=dialog_model)

# COMMAND ----------

# Register the model in MLflow Model Registry
run_id = "b3c787a275944d0f91332db8f83e446e"
mlflow.register_model(f"runs:/{run_id}/dialog_model", "DialogModel_Registry")

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd
import json

def create_tf_serving_json(data):
    return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
    url = 'https://adb-1012386050250820.0.azuredatabricks.net/serving-endpoints/Dialog_Model/invocations'
    headers = {'Authorization': f'Bearer dapi6912013219e5863b9be7d262dba4e1f3', 'Content-Type': 'application/json'}
    ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.request(method='POST', headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')

    return response.json()



# COMMAND ----------

import pandas as pd

# Prepare input data in a format that matches your model's requirements
input_data = pd.DataFrame({'sample': ['Hi']})  # Adjust 'input_column_name' and 'Hi' accordingly

try:
    # Call the score_model function to get predictions
    predictions = score_model("Hi, How are you doing?")
    
    # Handle the predictions based on your use case
    print("Model Predictions:")
    print(predictions)
except Exception as e:
    print(f"Error: {str(e)}")


# COMMAND ----------

dbutils.library.restartPython()
