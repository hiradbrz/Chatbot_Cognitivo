import gradio as gr
from model import QA_Model
from vectordb import vectorStore

qa_bot = QA_Model()
db = vectorStore()

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
        f"Gender: {gender}, ID: {user_id}, Age: {age}\n"
        "Feel free to ask me any questions you have.\n\n"
    )
    return welcome_message

def update_chat(history, user_input):
    if not user_input.strip():
        return history + "Bot: Please enter a question.\n\n", ""

    context = db.dbsearch(query=user_input)
    prompt = create_prompt(user_input, context)  
    bot_response = qa_bot.answer(prompt)

    return history + f"You: {user_input}\n\nBot: {bot_response}\n\n", ""

css_style = """
    body { font-family: Arial, sans-serif; }
    .gr-textbox, .gr-radio, .gr-number, .gr-button { width: 100%; }
    .gr-button { background-color: #4CAF50; color: white; margin-top: 10px; }
    .gr-output { margin-top: 15px; }
    .footer { text-align: center; margin-top: 20px; font-size: 0.8em; }
"""

def main_interface():
    with gr.Blocks(css=css_style) as block:
        gr.Markdown("🤖 NSW Land Tax Services Chatbot 💬")
        
        with gr.Row():
            gender = gr.Radio(["Male", "Female", "Other"], label="Gender")
            user_id = gr.Textbox(label="ID")
            age = gr.Number(label="Age", min=18, max=120)
            start_button = gr.Button("Start Chat")

        chat_history = gr.Textbox(label="Chat History", placeholder="Chat will appear here...", lines=15, interactive=False)
        user_message = gr.Textbox(label="Your Message", placeholder="Type your message here...")
        send_button = gr.Button("Send")

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
