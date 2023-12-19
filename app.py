import gradio as gr
from model import QA_Model
from vectordb import VectorStore

qa_bot = QA_Model()
db = VectorStore()

def create_prompt(question, context):
    """
    Create a prompt template for the QA model, specifically for handling
    inquiries about land tax and related problems for NSW service.
    """
    prompt = f"""
    As a knowledgeable assistant for New South Wales (NSW) land tax services, 
    answer the following question based on the provided context.

    Question: {question}

    Context Relevant to NSW Land Tax Services: {context}

    Answer:
    """
    return prompt.strip()

def start_chat(gender, user_id, age):
    welcome_message = (
        f"Hello! I'm here to assist you with NSW land tax services.\n"
        "Feel free to ask me any questions you have.\n\n"
    )
    return welcome_message

def update_chat(history, user_input):
    if not user_input.strip():
        return history + "Bot: Please enter a question.\n\n", ""

    # Searching for context
    context = db.db_search(query=user_input)

    # Handle empty or irrelevant inputs
    if not context:
        bot_response = "I'm not sure how to answer that. Could you provide more detail or ask a different question?"
    else:
        bot_response = qa_bot.answer(user_input, context)

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
