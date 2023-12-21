#%pip install torch
#%pip install mlflow transformers
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoTokenizer, AutoModelForCausalLM
import mlflow
from mlflow.pyfunc import PythonModel, PythonModelContext

class QA_Model:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        self.model = AutoModelForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def answer(self, question, context):
        # Split context into manageable chunks
        chunks = self.chunk_context(context)

        best_answer = None
        highest_confidence = -float('Inf')

        for chunk in chunks:
            inputs = self.tokenizer.encode_plus(question, chunk, add_special_tokens=True, return_tensors='pt', max_length=512, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            start_scores, end_scores = outputs.start_logits, outputs.end_logits
            answer_start = torch.argmax(start_scores)
            answer_end = torch.argmax(end_scores) + 1

            if answer_start >= answer_end or answer_end > inputs['input_ids'].size(1):  # Check for valid index range
                continue

            answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))

            confidence = self.calculate_confidence(start_scores, end_scores, answer_start, min(answer_end, inputs['input_ids'].size(1)))

            if confidence > highest_confidence:
                best_answer = answer
                highest_confidence = confidence

        return best_answer if best_answer is not None else "Sorry, I couldn't find an answer."

    def chunk_context(self, context, chunk_size=128):
        # Implement context chunking logic
        # This is a simplified example. You might need a more sophisticated approach for splitting the context.
        tokens = self.tokenizer.tokenize(context)
        for i in range(0, len(tokens), chunk_size):
            yield self.tokenizer.convert_tokens_to_string(tokens[i:i + chunk_size])


    def calculate_confidence(self, start_logits, end_logits, start_idx, end_idx):
        # Check if indices are within the range and adjust if necessary
        start_idx = min(start_idx, start_logits.size(1) - 1)
        end_idx = min(end_idx, end_logits.size(1) - 1)

        start_confidence = start_logits[0][start_idx]
        end_confidence = end_logits[0][end_idx - 1]  # end_idx is exclusive
        return (start_confidence + end_confidence).item() / 2


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
        # Assuming model_input is a string representing user input
        return super().generate_response_dia(model_input)

# Log the model in MLflow
with mlflow.start_run():
    dialog_model = DialogModelWrapper()
    mlflow.pyfunc.log_model("dialog_model", python_model=dialog_model)

# Register the model in MLflow Model Registry
# Replace 'your_run_id_here' with the actual run ID from MLflow
run_id = "your_run_id_here"
mlflow.register_model(f"runs:/{run_id}/dialog_model", "DialogModel_Registry")


class QA_ModelWrapper(QA_Model, PythonModel):
    def predict(self, context: PythonModelContext, model_input):
        # Assuming model_input is a dictionary with 'question' and 'context'
        return super().answer(model_input['question'], model_input['context'])

# Log the model in MLflow
with mlflow.start_run():
    qa_model = QA_ModelWrapper()
    mlflow.pyfunc.log_model("qa_model", python_model=qa_model)

# Register the model in MLflow Model Registry
# Replace 'your_run_id_here' with the actual run ID from MLflow
run_id = "your_run_id_here"
mlflow.register_model(f"runs:/{run_id}/qa_model", "QA_Model_Registry")

