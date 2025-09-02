from core_LLM import Chatmodel
from GPTpackages.ImageBufferMemory import encode_image
from MOBIpackages import ControlInterface
from utils import RAG, setup_logger, load_text
from tri_speech_packages.speech import audio_procession
import threading
import time
from pathlib import Path
from tri_speech_packages.trilingual_module import female_speak, minnan_speak2
from pprint import pprint
import queue
import os
import signal
from argparse import ArgumentParser
from distutils.util import strtobool
import asyncio
from tqdm import tqdm

def parse_args():
    parser=ArgumentParser()
    parser.add_argument("--speech",type=lambda x:bool(strtobool(x)),default=True,nargs='?',const=True,
                        help='determine the speech mode for recording the input query')
    parser.add_argument("--text",type=lambda x:bool(strtobool(x)),default=False,nargs='?',const=True,
                        help='determine the text mode for typing through keyboard to input the query')
    args=parser.parse_args()
    return args


def myinterruptspeak(language,interface):
    interface.state='speak'
    if language=='chinese':
        female_speak("抱歉，你想說什麼?",1,'fast','normal')
    if language=='english':
        female_speak("sorry, what are you talking about?",1,'fast','normal')
    if language=='taigi':
        minnan_speak2("抱歉，你想說什麼?")
    interface.state='idol'

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
MyAudio=audio_procession()
interface=ControlInterface.ControlInterface(enable_camera=False, show_img=False, enable_arm=True, enable_face=True, is_FullScreen=True)
print('hello')
#以上是啟用物件部分

def handle_exit(signum, frame):
    historydatabase.file_write()
    exit(0)

signal.signal(signal.SIGINT, handle_exit)
async def input_trans(MyAudio:object,user_input:str):
    task1 = asyncio.create_task(MyAudio.check_language(user_input))
    task2=asyncio.create_task(MyAudio.speech_to_text(user_input))
    responses = await asyncio.gather(task1,task2)
    print(responses)
    if responses[1]=='error':
        results = responses[0]
    else:
        results = (responses[0][0],responses[1],responses[0][2])
    return results


if __name__=="__main__":
    args=parse_args()

    #建立需要的參數
    text_dict={'what':''}
    interrupt=False
    action="nothing"
    stm=''
    for key_value in tqdm(keyworddocs):
        keywordbase.insert(key=key_value[0],value=key_value[1])

    #開始執行程式
    while True:
        if interrupt:#如果說話被打斷執行
            interrupt=False
            interface.state='speak'
            thread_1=threading.Thread(target=myinterruptspeak,args=(language,interface,),daemon=True)
            thread_1.start()
            time.sleep(1)
        print('hi')
        if not args.text:
            user_input=MyAudio.recording()#收音
            print("finish recording")
            if user_input=="None":#無聲音檔不執行
                continue
            start_time=time.perf_counter()
            #interface.get_frame()
            #fileList = os.listdir('input_img')
            fileList = []
            if fileList != []:
                img_list = [encode_image('input_img/' + fileList[-1])]

            language, text, emotion  = asyncio.run(input_trans(MyAudio,user_input))
        else:
            text = input("請輸入:")
            language = 'chinese'
        print("text: ",text)
        print("language: ", language)
        if text =="對話結束" or text =="對話結束。":
            break
        text_dict['what']=text
        text_dict['language']=language
        result,stm=mainLLM.run(text_dict)
        end_time1=time.perf_counter()
        #interrupt=MyAudio.speaking(result,language=language) #單獨播音(debug)
        interrupt=interface.express(result, emotion, 'nothing', language=language)#整合模型
        end_time2=time.perf_counter()
        elapsed_time1=end_time1-start_time
        elapsed_time2=end_time2-start_time
        print(emotion)
        print("generate_time",elapsed_time1)
        print("totaltime:", elapsed_time2)
    print('END!')

        
    

