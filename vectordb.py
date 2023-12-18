
import haystack
import os
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.pipelines import FAQPipeline
from haystack.utils import print_answers
import pandas as pd
ROOT = os.getcwd()
dataset_path = '\data\QA_samples.xlsx'

class vectorStore():
    def __init__(self)-> None:
        self.faiss_index_path = ROOT+'/vectorstore/my_faiss'
        print(self.faiss_index_path)
        if os.path.exists( self.faiss_index_path):
            self.document_store = FAISSDocumentStore.load(index_path= self.faiss_index_path, config_path=ROOT+'/vectorstore/my_config.json')
            self.retriever = EmbeddingRetriever(document_store=self.document_store, embedding_model='sentence-transformers/all-MiniLM-L6-v2', use_gpu=False)

        else:
            self.document_store = FAISSDocumentStore(
                        faiss_index_factory_str="Flat",
                        embedding_field="embedding",
                        embedding_dim=384,
                        similarity="cosine",
                        duplicate_documents="overwrite",  # using skip works, but the meta-data of the documents is not properly stored
                    )
            self.retriever = EmbeddingRetriever(document_store=self.document_store, embedding_model='sentence-transformers/all-MiniLM-L6-v2', use_gpu=False)
            self.createdb()

    def createdb(self):
        df = pd.read_excel(ROOT + dataset_path, index_col=None)

        df["Complaint"] = df["Complaint"].apply(lambda x: x.strip())
        complaints = list(df["Complaint"].values)
        df["embedding"] = self.retriever.embed_queries(queries=complaints).tolist()
        df = df.rename(columns={"Complaint": "content"})

        docs_to_index = df.to_dict(orient="records")
        self.document_store.write_documents(docs_to_index)
        self.document_store.update_embeddings(self.retriever)
        self.document_store.save(ROOT+'/vectorstore/my_faiss',config_path=ROOT+'/vectorstore/my_config.json')

    def dbsearch(self,query):
        pipe = FAQPipeline(retriever=self.retriever,)

        # Run any question and change top_k to see more or less answers
        prediction = pipe.run(query=query, params={"Retriever": {"top_k": 1}})
        df_predictions =pd.DataFrame(prediction['answers'])
        metadata_list =[]
        for i,row in df_predictions.iterrows():
            complaint = row['context']
            metadata_dict = (row['meta'])
            metadata_dict['Complaint'] = complaint
            metadata_list.append(metadata_dict)
        # df_predictions = pd.DataFrame(metadata_list,columns=['Complaint','Resolution','Feedback'])
        df_predictions = pd.DataFrame(metadata_list,columns=['Resolution'])
        
        predictions = df_predictions['Resolution'].values[0]
   
        return predictions

if __name__ == "__main__":
    # create_FAISS_db()
    db = vectorStore()
    db.createdb()
    df_predictions =db.dbsearch(query="I made the payment, but I was charged late payment fees")
    print(df_predictions)
   