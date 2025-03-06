import os 
from GPTpackages.GPTopenai import GPTopenai
from GPTpackages.LocalModel import local_LLM
from GPTpackages.PromptTemplate import PromptTemplate
from GPTpackages.ImageBufferMemory import ImageBufferMemory, encode_image
from GPTpackages.TextBuffer import TextBuffer
from pathlib import Path
import json
from opencc import OpenCC
import configparser

config=configparser.ConfigParser()
config.read('config.ini')
openai_key=config.get('openai','key1')

MODEL='gpt-4o-mini'
class Chatmodel():#核心對話模型
    def __init__(self,promptpath:str,knowledgeabase,memorybase,openai_key=openai_key,temperature=0.7)->None:
        self.key = openai_key
        self.stm=TextBuffer(buffer_size=5)#短期記憶，學長設計的
        self.kdb=knowledgeabase
        self.mdb=memorybase
        self.trans=OpenCC('s2twp')#簡中轉繁中用，備用的
        self.model=local_LLM(
            prompt=promptpath,
            temperature=temperature,
            img_memory=ImageBufferMemory())
    def text_retrival(self,text,keyword) -> str:#取出收音的文字
        if keyword in self.title:#偵測到關鍵字啟用
            dataretiver=self.local_db.get_retriver(keyword=keyword)
        else:
            dataretiver=self.local_db.get_retriver()
        results=dataretiver.invoke(text)
        output=''
        for i,content in enumerate(results):
            output=output+f"Document {i+1}: Text:{content.page_content}\n\
                            Metadata: {content.metadata}\n"

        #print(output)

        return output
    def run_slot(self,texts:dict,stm:str)->dict:#intention model用
        mytexts={}
        mytexts['what']=texts['what']
        mytexts['context']=stm
        output=self.model.run(text_dict=mytexts)
        print(output)
        result=json.loads(output)
        return result
    def run_intention(self,texts:dict,stm:str)->dict:#intention model用
        mytexts={}
        mytexts['what']=texts['what']
        mytexts['context']=stm
        result=self.model.run(text_dict=mytexts)
        return result
    def run(self,texts:dict)->list: #主對話model用
        # texts['intention']=intent
        # texts['preacher']=slot['preacher'] if slot['preacher']!=None else ''
        # rag_text=f"解經段落:{slot['paragraph']}，提問: "+texts['what'] if slot['preacher']!=None else texts['what']
        # texts["local_data"]=self.text_retrival(rag_text,slot['preacher'])
        # conversation=self.ltm.retrive_text(rag_text,inf_num=2)
        # texts["context"]=self.stm.get()
        # texts["long_memory"]=conversation
        # texts["paragraph"]=slot["paragraph"] if slot['paragraph']!=None else ''
        #以上是撈本地資料、stm、ltm各種資料
        # if slot['image_request']==False:
        #     img_list=[]       
        knowldge=""
        memory=''
        texts["context"]=self.stm.get()
        top_k_knowledge=self.kdb.retrieve(texts["what"],5)
        top_k_memory=self.mdb.retrieve(texts["what"],5)
        print(top_k_knowledge)
        print(top_k_memory)
        for i,content in enumerate(top_k_knowledge):
            knowldge=knowldge+f"Document {i+1}: Text:{content}\n"
        texts["local_data"]=knowldge
        for i,content in enumerate(top_k_memory):
            memory=memory+f"Document {i+1}: Text:{content}\n"
        texts["long_memory"]=memory
        output=self.model.run(text_dict=texts,img_list=[],img_refresh=True)#丟給GPT
        if texts['language']=='chinese':
            input=self.trans.convert(texts['what'])
        else:
            input=texts['what']
        output=self.trans.convert(output)
        input=texts['what']  
        with open('./cv.txt',mode='a',encoding='utf-8')as f:
            f.write(f'Human:{input}\nMobi:{output}\n')
        results=[input,output]
        self.stm.set(results)
        return results,texts["context"]+self.stm.get(1)