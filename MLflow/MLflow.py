import os
import pandas as pd
import logging
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.pipelines import FAQPipeline
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import mlflow
import gradio as gr

# Initialize MLflow
mlflow_tracking_uri="/Users/hirad/Chatbot_Cognitivo-2/MLflow"
mlflow.set_tracking_uri(mlflow_tracking_uri)

experiment_name = "QA_Model_Experiment"
if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(experiment_name)

mlflow.set_experiment(experiment_name)

class VectorStore:
    def __init__(self):
        self.root = os.getcwd()
        self.dataset_path = os.environ.get('DATASET_PATH', '/data/QA_LandTax.xlsx')
        self.faiss_index_path = os.path.join(self.root, 'vectorstore', 'my_faiss')

        embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'
        embedding_dim = 384
        similarity = "cosine"

        if os.path.exists(self.faiss_index_path):
            self.document_store = FAISSDocumentStore.load(index_path=self.faiss_index_path, config_path=os.path.join(self.root, 'vectorstore', 'my_config.json'))
        else:
            self.document_store = FAISSDocumentStore(
                faiss_index_factory_str="Flat",
                embedding_field="embedding",
                embedding_dim=embedding_dim,
                similarity=similarity,
                duplicate_documents="overwrite"
            )

        self.retriever = EmbeddingRetriever(document_store=self.document_store, embedding_model=embedding_model, use_gpu=False)
        self.pipe = FAQPipeline(retriever=self.retriever)

        # Log parameters
        mlflow.log_param("embedding_model", embedding_model)
        mlflow.log_param("embedding_dim", embedding_dim)
        mlflow.log_param("similarity", similarity)

        if not os.path.exists(self.faiss_index_path):
            self.create_db()

class QA_Model:
    def __init__(self):
        model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Log model and device details
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("computation_device", str(self.device))

    def chunk_context(self, context, chunk_size=128):
        tokens = self.tokenizer.tokenize(context)
        for i in range(0, len(tokens), chunk_size):
            yield self.tokenizer.convert_tokens_to_string(tokens[i:i + chunk_size])

    def calculate_confidence(self, start_logits, end_logits, start_idx, end_idx):
        start_idx = min(start_idx, start_logits.size(1) - 1)
        end_idx = min(end_idx, end_logits.size(1) - 1)

        start_confidence = start_logits[0][start_idx]
        end_confidence = end_logits[0][end_idx - 1]
        return (start_confidence + end_confidence).item() / 2

    def answer(self, question, context):
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

            if answer_start >= answer_end or answer_end > inputs['input_ids'].size(1):
                continue

            answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
            confidence = self.calculate_confidence(start_scores, end_scores, answer_start, answer_end)

            if confidence > highest_confidence:
                best_answer = answer
                highest_confidence = confidence

        return best_answer if best_answer else "Sorry, I couldn't find an answer."


qa_bot = QA_Model()
db = VectorStore()

def start_chat(gender, user_id, age):
    welcome_message = "Hello! I'm here to assist you with NSW land tax services.\nFeel free to ask me any questions you have.\n\n"
    
    # Log user information (anonymously)
    mlflow.log_param("user_id", user_id)
    mlflow.log_param("user_age", age)
    mlflow.log_param("user_gender", gender)
    
    return welcome_message

def update_chat(history, user_input):
    start_time = time.time()
    
    if not user_input.strip():
        return history + "Bot: Please enter a question.\n\n", ""

    context = db.db_search(query=user_input)

    if not context:
        bot_response = "I'm not sure how to answer that. Could you provide more detail or ask a different question?"
    else:
        bot_response = qa_bot.answer(user_input, context)

    response_time = time.time() - start_time
    
    # Log the response time and user input
    mlflow.log_metric("response_time", response_time)
    mlflow.log_param("user_input", user_input)
    mlflow.log_param("bot_response", bot_response)

    return history + f"You: {user_input}\n\nBot: {bot_response}\n\n", ""

css_style = """

    /* existing CSS */
    .loading { animation: spin 1s linear infinite; }
    @keyframes spin { 100% { transform: rotate(360deg); } }

    /* Basic reset */
    * { box-sizing: border-box; }
    body, html { 
        margin: 0; padding: 0; font-family: 'Arial', sans-serif; 
        background-color: #1E1E1E; color: #FFFFFF;
    }

    /* Container styling */
    .gradio-container { 
        max-width: 700px; margin: 30px auto; padding: 20px; 
        background-color: #2D2D2D; border-radius: 8px; 
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2); 
    }

    /* Header styling */
    h1 { color: #FFF; text-align: center; margin-bottom: 30px; }

    /* Input fields styling */
    .gr-textbox, .gr-number, .gr-radio { width: 100%; margin-bottom: 15px; }
    .gr-textbox { 
        border-radius: 4px; border: 1px solid #555; padding: 10px 15px; 
        font-size: 16px; background-color: #333; color: #FFF;
    }
    .gr-textbox:focus { outline: none; border-color: #4A90E2; }
    .gr-radio label { margin-right: 15px; color: #FFF; }

    /* Button styling */
    .gr-button { 
        background-color: #4A90E2; color: white; padding: 10px 20px; 
        border: none; border-radius: 4px; cursor: pointer; 
        font-size: 16px; margin-top: 10px; 
    }
    .gr-button:hover { background-color: #357ABD; }

    /* Chat history styling */
    .chat-history { 
        height: 300px; overflow-y: auto; background-color: #333; 
        border-radius: 4px; border: 1px solid #555; 
        padding: 10px; margin-bottom: 15px; color: #FFF;
    }

    /* Chat message styling */
    .user-message, .bot-message { 
        border-radius: 15px; padding: 8px 12px; 
        max-width: 80%; margin-bottom: 5px; word-wrap: break-word;
    }
    .user-message { 
        background-color: #5cb85c; margin-left: auto; 
        margin-right: 10px; text-align: right;
    }
    .bot-message { 
        background-color: #337ab7; margin-left: 10px; 
        margin-right: auto; text-align: left;
    }

    /* Footer styling */
    .footer { 
        text-align: center; padding: 15px 0; background-color: #333; 
        color: white; font-size: 14px; margin-top: 20px; 
        border-radius: 0 0 8px 8px; 
    }
"""

def main_interface():
    with gr.Blocks(css=css_style) as block:
        gr.Markdown("🤖 NSW Land Tax Services Chatbot 💬")

        with gr.Row():
            gender = gr.Radio(["Male", "Female", "Other"], label="Gender")
            user_id = gr.Textbox(label="ID", placeholder="Enter your ID", elem_id="user_id")
            age = gr.Number(label="Age", min=18, max=120, elem_id="age")
            start_button = gr.Button("Start Chat")

        chat_history = gr.Textbox(label="Chat History", placeholder="Chat will appear here...", lines=15, interactive=False)
        user_message = gr.Textbox(label="Your Message", placeholder="Type your message here...")
        send_button = gr.Button("Send", elem_id="send_button")

        start_button.click(start_chat, inputs=[gender, user_id, age], outputs=chat_history)
        send_button.click(update_chat, inputs=[chat_history, user_message], outputs=[chat_history, user_message])

        gr.Markdown('''
            <div class="footer">
                <p>Powered by Cognitivo</p>
            </div>
        ''')

    return block

if __name__ == "__main__":
    main_interface().launch(share=False)