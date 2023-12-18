from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
from ctransformers import AutoModelForCausalLM
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import torch

import sys
import psutil
import os
from pathlib import Path


ROOT = os.getcwd()

DB_FAISS_PATH = ROOT + '/vectorstore/db_faiss'
DB_CHROMA_PATH = ROOT + '/vectorstore/db_chroma'


class summary_llm():
    def __init__(self,task='summarization') -> None:
        self.summary_checkpoint = "MBZUAI/LaMini-Flan-T5-783M"
        self.summarytokenizer = T5Tokenizer.from_pretrained(self.summary_checkpoint)
        self.summarybase_model = T5ForConditionalGeneration.from_pretrained(self.summary_checkpoint, device_map='auto', torch_dtype=torch.float32)
        self.summary_pipe = pipeline(
            'summarization',
            model = self.summarybase_model,
            tokenizer = self.summarytokenizer,
            max_length = 1000, 
            min_length = 50)
        # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
    
            

