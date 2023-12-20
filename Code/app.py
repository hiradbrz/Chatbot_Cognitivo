import gradio as gr
from Code.model import DialogModel, QA_Model
from Code.vectordb import VectorStore

# Initialize DialogModel and QA_Model
dialog_bot = DialogModel()
qa_bot = QA_Model()
db = VectorStore()

def create_prompt(question, context, query_type):
    """
    Create a prompt template based on the query type.
    """
    if query_type == "TaxGPT":
        # Prompt template for LanTax queries
        prompt = f"""
        As a knowledgeable assistant for New South Wales (NSW) land tax services, 
        answer the following question based on the provided context.
        Also show the context details in the output, because it makes the bot better.

        Question: {question}

        Context Relevant to NSW Land Tax Services: {context}

        Answer:
        """
    else:
        # Prompt template for General queries
        prompt = f"""
        You've asked a general question. Please provide more details if necessary.

        Question: {question}

        Context: {context}

        Answer:
        """
    return prompt.strip()


def start_chat(gender, user_id, age, query_type):
    welcome_message = (
        f"Hello! I'm here to assist you with NSW land tax services.\n"
        "Feel free to ask me any questions you have.\n\n"
    )
    prompt = create_prompt("", "", query_type)  # Generate the prompt based on query_type
    return welcome_message, query_type

def update_chat(history, user_input, query_type):
    if not user_input.strip():
        return history + "Bot: Please enter a message.\n\n", ""

    if query_type == "General":
        # Use DialogModel for general queries
        bot_response = dialog_bot.generate_response_dia(user_input)
        bot_response_message = "Bot (DialogModel):"
    elif query_type == "TaxGPT":
        # Use QA_Model for TaxGPT queries
        context = db.db_search(query=user_input)
        if context:
            bot_response = qa_bot.answer(user_input, context)
            context_details = f"Context: {context}\n\n"
            bot_response = context_details + bot_response
            bot_response_message = "Bot (QA_Model):"
        else:
            bot_response = "Sorry, I couldn't find relevant information for your enquiry."
            bot_response_message = "Bot (QA_Model):"
    else:
        bot_response = "Invalid query type selected."
        bot_response_message = "Bot:"

    return history + f"You: {user_input}\n\n{bot_response_message} {bot_response}\n\n", ""


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
            query_type = gr.Radio(["General", "TaxGPT"], label="Select Query Type")
            user_id = gr.Textbox(label="ID", placeholder="Enter your ID", elem_id="user_id")
            age = gr.Number(label="Age", min=18, max=120, elem_id="age")
            start_button = gr.Button("Start Chat")

        chat_history = gr.Textbox(label="Chat History", placeholder="Chat will appear here...", lines=15, interactive=False)
        user_message = gr.Textbox(label="Your Message", placeholder="Type your message here...")
        send_button = gr.Button("Send", elem_id="send_button")

        start_button.click(start_chat, inputs=[user_id, age, query_type], outputs=[chat_history, query_type])
        send_button.click(update_chat, inputs=[chat_history, user_message, query_type], outputs=[chat_history, user_message])

        gr.Markdown('''
            <div class="footer">
                <p>Powered by Cognitivo</p>
            </div>
        ''')

    return block  # Return the Gradio interface block

if __name__ == "__main__":
    main_interface().launch(share=False)
