import faiss
from pprint import pprint
import torch
from pathlib import Path
from enum import Enum
from datasets import load_dataset
import logging
import json
from typing import Optional
from transformers import AutoModel, AutoTokenizer
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import random
import numpy as np

class JSONLinesHandler(logging.FileHandler):
    def emit(self, record):
        log_entry = self.format(record)
        with open(self.baseFilename, 'a') as file:
            file.write(f"{log_entry}\n")
            
def setup_logger(name, log_file, level=logging.INFO):
    """Function to set up jsonlines logger."""
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, 'w') as file:
        pass  # create the file if it does not exist

    formatter = logging.Formatter('[%(asctime)s][%(name)-5s][%(levelname)-5s] %(message)s')  # Only message gets logged
    handler = JSONLinesHandler(log_file)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

class RetrieveOrder(Enum):
    SIMILAR_AT_TOP = "similar_at_top"  # the most similar retrieved chunk is ordered at the top
    SIMILAR_AT_BOTTOM = "similar_at_bottom"  # reversed
    RANDOM = "random"  # randomly shuffle the retrieved chunks


class RAG():
    def __init__(self,rag_config:Optional[dict]=None)->None:
        if rag_config==None:
            rag_config = {
                "embedding_model": "thenlper/gte-base",#BAAI/bge-base-en-v1.5
                "rag_filename": "test_rag_pool",
                "seed": 42,
                "top_k": 5,
                "order": "similar_at_top"  # ["similar_at_top", "similar_at_bottom", "random"]
            }
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"模型將運行在: {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(rag_config["embedding_model"])
        self.embed_model = AutoModel.from_pretrained(rag_config["embedding_model"]).eval()

        self.index = None
        self.id2evidence = dict()
        self.embed_dim = len(self.encode_data("Test embedding size"))
        self.insert_acc = 0

        self.seed = rag_config["seed"]
        self.top_k = rag_config["top_k"]
        orders = {member.value for member in RetrieveOrder}
        assert rag_config["order"] in orders
        self.retrieve_order = rag_config["order"]
        random.seed(self.seed)
        # TODO: make a file to save the inserted rows
    def debug_check(self):
        query="可以跟我說說心經的內容嗎?"
        print(self.insert_acc)
        results = self.retrieve(query, top_k=10)
        pprint(results)
    def create_faiss_L2index(self):
        self.index=faiss.IndexFlatL2(self.embed_dim)
    def create_faiss_INFPQindex(self,key_value_pairs):
        # Create a FAISS index
        nlist=50
        M=16
        n_bits=4
        self.quantizer=faiss.IndexFlatL2(self.embed_dim)
        self.index = faiss.IndexIVFPQ(self.quantizer,self.embed_dim,nlist,M,n_bits)
        key_list=[]
        value_list=[]
        for keys,values in key_value_pairs:
            key_list.append(keys)
            value_list.append(values)
        self.index_initialize(keylist=key_list,valuelist=value_list)
    def encode_data(self, sentence: str) -> np.ndarray:
        # Tokenize the sentence
        encoded_input = self.tokenizer([sentence], padding=True, truncation=True, return_tensors="pt")
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.embed_model(**encoded_input)
            # Perform pooling. In this case, cls pooling.
            sentence_embeddings = model_output.last_hidden_state[:, 0]
        feature = sentence_embeddings.numpy()[0]
        norm = np.linalg.norm(feature)
        return feature / norm
    def index_initialize(self,keylist:list,valuelist:list)->None:
        xb=np.zeros((1,1))
        if Path('index_file.index').exists():
            self.index=faiss.read_index('index_file.index')
            self.insert_acc=self.index.ntotal
            with open("id2text.json", "r") as f:
                self.id2evidence = json.load(f)
            assert self.index.is_trained
        else:
            for key,value in zip(keylist,valuelist):
                key_embedding=np.expand_dims(self.encode_data(key).astype('float32'),axis=0)
                if not xb.any():
                    xb=key_embedding
                else:
                    xb=np.concatenate((xb,key_embedding),axis=0)
            
            self.id2evidence={str(i):value for i,value in enumerate(valuelist)}
            with open('id2text.json',"w") as f:
                json.dump(self.id2evidence,f) 
            assert not self.index.is_trained
            self.index.train(xb)
            assert self.index.is_trained
            self.index.add(xb)
            self.insert_acc=self.index.ntotal-1
            faiss.write_index(self.index,'index_file.index')


    def insert(self, key:str, value:str)->None:
        """Use the key text as the embedding for future retrieval of the value text."""
        embedding = self.encode_data(key).astype('float32')  # Ensure the data type is float32
        self.index.add(np.expand_dims(embedding, axis=0))
        self.id2evidence[str(self.insert_acc)] = value
        self.insert_acc += 1

    def retrieve(self, query: str, top_k: int) -> list[str]:
        """Retrieve top-k text chunks"""
        embedding = self.encode_data(query).astype('float32')  # Ensure the data type is float32
        top_k = min(top_k, self.insert_acc)
        distances, indices = self.index.search(np.expand_dims(embedding, axis=0), top_k)
        distances = distances[0].tolist()
        indices = indices[0].tolist()
        
        results = [{'link': str(idx), '_score': {'faiss': dist}} for dist, idx in zip(distances, indices)]
        # Re-order the sequence based on self.retrieve_order
        if self.retrieve_order == RetrieveOrder.SIMILAR_AT_BOTTOM.value:
            results = list(reversed(results))
        elif self.retrieve_order == RetrieveOrder.RANDOM.value:
            random.shuffle(results)
        
        text_list = [self.id2evidence[result["link"]] for result in results]
        return text_list    
    
def load_text(path:str,spilitter:Optional[RecursiveCharacterTextSplitter]=None)-> list[tuple]:
        texts=[]
        sep=["\n\n","\n"," ", "\u200b",  # Zero-width space
        "\uff0c",  # Fullwidth comma
        "\u3001",  # Ideographic comma
        "\uff0e",  # Fullwidth full stop
        "\u3002",  # Ideographic full stop
        "",]
        if spilitter==None: 
            splitter=RecursiveCharacterTextSplitter(separators=sep,chunk_size=128,chunk_overlap=8)
        raw_documents=None
        assert isinstance(path, str)
        for content in Path(path).glob("*.txt"):
            raw_documents = TextLoader(str(content), encoding='utf-8').load_and_split(splitter)
            if path=="./scripts":#讀取經文資料時對metadata進行處理並儲存#注意linux和windows差距
                for name in raw_documents:
                    # print(name)
                    title=name.metadata["source"].split("_")[0]
                    print(title)
                    title= title.split("/")[1]
                    key="source: "+ title +", content:" +name.page_content
                    value=name.page_content
                    texts.append((key,value))                    
            else:
                for name in raw_documents:
                    key=name.page_content
                    value=name.page_content
                    texts.append((key,value))
        return texts
    
# def get_SQUAD():
# def test():
if __name__=="__main__":
    rag_config = {
                "embedding_model": "thenlper/gte-base",#BAAI/bge-base-en-v1.5
                "rag_filename": "test_rag_pool",
                "seed": 42,
                "top_k": 5,
                "order": "similar_at_top"  # ["similar_at_top", "similar_at_bottom", "random"]
            }

    rag = RAG(rag_config=rag_config)
    # Key-value pairs for testing
    key_value_pairs = [
        ("Apple is my favorite fruit", "Oh really?"),
        ("What is your favorite fruit?", "Lettuce, tomato, and spinach."),
        ("What is your favorite vegetable?", "Apple, banana, and watermelon."),
        ("What do you like to read in your free time?", "Sherlock Holmes")
    ]

    # Insert the key-value pairs into the RAG
    
    for key, value in key_value_pairs:
        rag.index_initialize(key, key + ' ' + value)


    query = "I like to eat lettuce."
    results = rag.retrieve(query, top_k=rag_config["top_k"])
    pprint(results)
    