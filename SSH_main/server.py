from core_LLM import Chatmodel
from GPTpackages.ImageBufferMemory import encode_image
from utils import RAG, setup_logger, load_text
from TCPpackages import SocketServer
from pprint import pprint
import asyncio
from tqdm import tqdm
import signal
import re

rag_config = {
                "embedding_model": "Qwen/Qwen3-Embedding-0.6B",#BAAI/bge-base-en-v1.5#盡量找BERT類型的Model
                "rag_filename": "test_rag_pool",
                "seed": 42,
                "top_k": 5,
                "order": "similar_at_top"  # ["similar_at_top", "similar_at_bottom", "random"]
            }#RAG設定
setup_logger(__name__,"main_output.log")#開啟日誌

#以下是啟用各項物件
knowledgebase=RAG("kdb",rag_config=rag_config)#載入RAG database
knowledgedocs=load_text(path=".\scripts")#載入文檔#list[tuple] 
"""
目前問題:文件讀不進去，應該是卡在load_text
解決了，問題是linux和windows的路徑分隔符(/和\的問題)
"""
knowledgebase.create_faiss_INFPQindex(knowledgedocs)
keywordbase=RAG("keyword",rag_config=rag_config)
keyworddocs=load_text(path=".\scripts\scripture_content",chunk_size=24,chunk_overlap=0)
keywordbase.create_faiss_L2index()
historydatabase=RAG("mdb",rag_config=rag_config)
historydatabase.create_faiss_L2index()
mainLLM=Chatmodel(promptpath='.\prompts\chat_prompt.txt',knowledgeabase=knowledgebase,memorybase=historydatabase,keywordbase=keywordbase)

def handle_exit(signum, frame):
    historydatabase.file_write()
    exit(0)
signal.signal(signal.SIGINT, handle_exit)

sen = SocketServer(host_ip='140.112.14.248', port=12345)
rec = SocketServer(host_ip='140.112.14.248', port=12345)
if __name__=="__main__":
    for key_value in tqdm(keyworddocs):
        keywordbase.insert(key=key_value[0],value=key_value[1])
    while True:
        text = rec.wait_msg()
        print(text)