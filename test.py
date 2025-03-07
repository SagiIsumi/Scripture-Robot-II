from core_LLM import Chatmodel
from GPTpackages.ImageBufferMemory import encode_image
from MOBIpackages import ControlInterface
from utils import RAG, setup_logger, load_text
import speech
import threading
import time
from pathlib import Path
from trilingual_module import female_speak, minnan_speak2
from pprint import pprint
import queue
import os

def myinterruptspeak(language,interface):
    interface.state='speak'
    if language=='chinese':
        female_speak("抱歉，你想說什麼?",1,'fast','normal')
    if language=='english':
        female_speak("sorry, what do you say?",1,'fast','normal')
    if language=='taigi':
        minnan_speak2("抱歉，你想說什麼?")
    interface.state='idol'

if __name__=="__main__":
    rag_config = {
                "embedding_model": "thenlper/gte-base",#BAAI/bge-base-en-v1.5#盡量找BERT類型的Model
                "rag_filename": "test_rag_pool",
                "seed": 42,
                "top_k": 5,
                "order": "similar_at_top"  # ["similar_at_top", "similar_at_bottom", "random"]
            }#RAG設定
    setup_logger("main","main_output.log")#開啟日誌
    #以下是啟用各項物件
    knowledgebase=RAG(rag_config=rag_config)#載入RAG database
    key_value_pairs=load_text(path=".\scripts")#載入文檔
    knowledgebase.create_faiss_INFPQindex(key_value_pairs)
    historydatabase=RAG(rag_config=rag_config)
    historydatabase.create_faiss_L2index()
    mainLLM=Chatmodel(promptpath='.\prompts\chat_prompt.txt',knowledgeabase=knowledgebase,memorybase=historydatabase)
    MyAudio=speech.audio_procession()
    interface=ControlInterface.ControlInterface(enable_camera=False, show_img=False, enable_arm=False, enable_face=False, is_FullScreen=False)
    #以上是啟用物件部分

    #建立需要的參數
    text_dict={'what':''}
    interrupt=False
    action="nothing"
    stm=''

    #開始執行程式
    while True:
        if interrupt:#如果說話被打斷執行
            interrupt=False
            interface.state='speak'
            thread_1=threading.Thread(target=myinterruptspeak,args=(language,interface,),daemon=True)
            thread_1.start()
            time.sleep(1)
        input=MyAudio.recording()#收音
        if input=="None":#無聲音檔不執行
            continue
        interface.get_frame()
        fileList = os.listdir('input_img')
        if fileList != []:
            img_list = [encode_image('input_img/' + fileList[-1])]
        language=thread_language=threading.Thread(target=MyAudio.check_language,args=input)
        text=thread_transcript=threading.Thread(target=MyAudio.speech_to_text,args=input)
        thread_language.join()
        thread_transcript.join()
        text_dict['what']=text
        text_dict['language']=language
        result,stm=mainLLM.run(text_dict)
        print(result)
        interrupt=MyAudio.speaking(result[1],language=language)

