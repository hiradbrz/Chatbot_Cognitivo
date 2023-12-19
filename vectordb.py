
import haystack
import os
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.pipelines import FAQPipeline
from haystack.utils import print_answers
import pandas as pd
ROOT = os.getcwd()
dataset_path = "/data/QA_LandTax.xlsx"
print(ROOT+dataset_path)

class vectorStore():
    def __init__(self)-> None:
        self.faiss_index_path = ROOT+'/vectorstore/my_faiss'
        print(self.faiss_index_path)

        # Initialize the document store
        if os.path.exists(self.faiss_index_path):
            self.document_store = FAISSDocumentStore.load(index_path=self.faiss_index_path, config_path=ROOT+'/vectorstore/my_config.json')
        else:
            self.document_store = FAISSDocumentStore(
                faiss_index_factory_str="Flat",
                embedding_field="embedding",
                embedding_dim=384,
                similarity="cosine",
                duplicate_documents="overwrite"
            )

        # Initialize the retriever
        self.retriever = EmbeddingRetriever(document_store=self.document_store, embedding_model='sentence-transformers/all-MiniLM-L6-v2', use_gpu=False)

        # Initialize the FAQPipeline
        self.pipe = FAQPipeline(retriever=self.retriever)

        # If the document store is newly created, populate it with data
        if not os.path.exists(self.faiss_index_path):
            self.createdb()

    def dbsearch(self, query):
        # Run the query through the FAQPipeline
        prediction = self.pipe.run(query=query, params={"Retriever": {"top_k": 1}})
        
        # Print the entire prediction for debugging
        print("Full prediction:", prediction)

        # Check if any answers are found
        if prediction['answers']:
            # Attempt to extract the 'answer' from the correct location
            try:
                answer = prediction['answers'][0].meta['answer']
            except KeyError:
                # If 'answer' key is not found, print the answer object for further inspection
                print("Answer object:", prediction['answers'][0])
                answer = "Sorry, I couldn't find an answer to your question."
        else:
            answer = "Sorry, I couldn't find an answer to your question."

        return answer   
    def createdb(self):
        # Load the dataset
        df = pd.read_excel(ROOT + dataset_path, index_col=None)

        # Assume the first column is 'Question' and the second is 'Answer'
        df.columns = ['Question', 'Answer']

        # Preprocess the data
        df["Question"] = df["Question"].apply(lambda x: x.strip())
        df["Answer"] = df["Answer"].apply(lambda x: x.strip())

        # Embedding the questions for retrieval
        df["embedding"] = self.retriever.embed_queries(queries=df["Question"].tolist()).tolist()

        # Prepare the data for indexing in FAISSDocumentStore
        docs_to_index = [{
            'content': row['Question'],
            'meta': {'answer': row['Answer']}
        } for _, row in df.iterrows()]

        # Write documents to the document store and update embeddings
        self.document_store.write_documents(docs_to_index)
        self.document_store.update_embeddings(self.retriever)
        self.document_store.save(ROOT+'/vectorstore/my_faiss', config_path=ROOT+'/vectorstore/my_config.json')


if __name__ == "__main__":
    # create_FAISS_db()
    db = vectorStore()
    db.createdb()
    df_predictions =db.dbsearch(query="What is property tax")
    print(df_predictions)
   