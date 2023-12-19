from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import pipeline
import torch

class QA_Model():
    def __init__(self):
        # Load pre-trained model tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained('deepset/roberta-base-squad2')
        self.model = AutoModelForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2')

        # Check if GPU is available and use it
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def answer(self, question, context):
        # Combine question and context for the model
        inputs = self.tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract the start and end tokens with the highest probability of being the answer
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1

        # Convert tokens to the answer string
        answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))

        return answer
