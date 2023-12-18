import gradio as gr
import re

# from model import qa_bot
from model import summary_llm
from vectordb import vectorStore

import time

summary_bot = summary_llm(task='summarization')
vstore = vectorStore()


def edit_text(text,lang_str):
    return text,lang_str


def summarize(text,option,feedbackoption):
    if text == None : return ""
    time.sleep(1)
    
    input_prompt =f"""
            You are a customer service officer who listen to customer enquiries and complaints.
            Write a concise and short summary of the following customer enquiry.Do not add resolution steps.
            ```{text}```
        """
    result = summary_bot.summary_pipe(input_prompt)  
    result = result[0]['summary_text']
    if option == "Customer Enquiry":
        search_results = vstore.dbsearch(query=text)

    return result,search_results

title = """🎤 Customer Service Bot 💬"""

custom_css = """
  #banner-image {
    display: block;
    margin-left: auto;
    margin-right: auto;
  }
  #chat-message {
    font-size: 14px;
    min-height: 300px;
  }
"""


block = gr.Blocks(css=custom_css)

with block:
    gr.HTML(title)

    with gr.Group():
        with gr.Box():
       
            # Transcribe Button
            text = gr.Textbox(label="Transcription",)
            with gr.Row(): 
                submit_btn = gr.Button("Submit",scale=0)
                clear_btn = gr.Button("Clear",scale=0)
                
            with gr.Box(): 
                # query Option
                query_option = gr.Radio(
                choices=["Customer Enquiry", "General"],
                label="Select an option",
                default="Customer Enquiry"
                )
                print(query_option)
            
                summary_output =  gr.Textbox(label="Summary")
                summary_btn = gr.Button("Summary")
                with gr.Row():
                    qa_history = gr.Textbox(label="Resolution")
                    feedback_option = gr.Radio(
                    choices=["👍", "👎"],
                    label="Was this recommendation useful?",
                    default = "👍"
                    )

                
                submit_btn.click(
                edit_text,
                inputs=[
                    text                    
                ],
                outputs=[
                    text
                   
                ]
                )
               
                summary_btn.click(
                summarize,
                inputs=[
                    text,
                    query_option,
                    feedback_option         
                ],
                outputs=[
                    summary_output,
                 
                    qa_history
                ]
                )
               
           
        gr.HTML('''
        <div class="footer">
            <p>Demo by Cognitivo
            </p>
        </div>
        ''')

block.launch(share=False,server_name="0.0.0.0")







