import os
import pandas as pd
import logging
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.pipelines import FAQPipeline

# Setup basic logging
logging.basicConfig(level=logging.INFO)

class VectorStore:
    def __init__(self):
        self.root = os.getcwd()
        self.dataset_path = os.environ.get('DATASET_PATH', '/data/QA_LandTax.xlsx')
        self.faiss_index_path = os.path.join(self.root, 'vectorstore', 'my_faiss')

        # Initialize or load the document store
        if os.path.exists(self.faiss_index_path):
            self.document_store = FAISSDocumentStore.load(index_path=self.faiss_index_path, config_path=os.path.join(self.root, 'vectorstore', 'my_config.json'))
        else:
            self.document_store = FAISSDocumentStore(
                faiss_index_factory_str="Flat",
                embedding_field="embedding",
                embedding_dim=384,
                similarity="cosine",
                duplicate_documents="overwrite"
            )

        # Initialize the retriever and pipeline
        self.retriever = EmbeddingRetriever(document_store=self.document_store, embedding_model='sentence-transformers/all-MiniLM-L6-v2', use_gpu=False)
        self.pipe = FAQPipeline(retriever=self.retriever)

        if not os.path.exists(self.faiss_index_path):
            self.create_db()

    def db_search(self, query):
        try:
            prediction = self.pipe.run(query=query, params={"Retriever": {"top_k": 1}})
            logging.info("Full prediction: %s", prediction)

            if prediction['answers']:
                try:
                    return prediction['answers'][0].meta['answer']
                except KeyError:
                    logging.error("Answer key not found in the response: %s", prediction['answers'][0])
                    return "Sorry, I couldn't find an answer to your question."
            else:
                return "Sorry, I couldn't find an answer to your question."
        except Exception as e:
            logging.error("Error during document search: %s", str(e))
            return "An error occurred while processing your query."

    def create_db(self):
        try:
            df = pd.read_excel(os.path.join(self.root, self.dataset_path))
            df.columns = ['Question', 'Answer']
            df['Question'] = df['Question'].apply(lambda x: x.strip())
            df['Answer'] = df['Answer'].apply(lambda x: x.strip())
            df['embedding'] = self.retriever.embed_queries(queries=df['Question'].tolist()).tolist()

            docs_to_index = [{'content': row['Question'], 'meta': {'answer': row['Answer']}} for _, row in df.iterrows()]
            self.document_store.write_documents(docs_to_index)
            self.document_store.update_embeddings(self.retriever)
            self.document_store.save(os.path.join(self.root, 'vectorstore', 'my_faiss'), config_path=os.path.join(self.root, 'vectorstore', 'my_config.json'))

            logging.info("Database created successfully.")
        except Exception as e:
            logging.error("Error during database creation: %s", str(e))

