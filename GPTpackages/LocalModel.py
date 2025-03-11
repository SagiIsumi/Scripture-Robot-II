from openai import OpenAI
from GPTpackages.PromptTemplate import PromptTemplate
from vllm import LLM, SamplingParams
import configparser

class local_LLM():
    def __init__(self,prompt:str,model:str="yentinglin/Llama-3-Taiwan-8B-Instruct-awq",
                 temperature:int=1, img_memory=None):
        myconfig=configparser.ConfigParser()
        myconfig.read('config.ini')
        APIKEY=myconfig.get('openai','key1')
        openai_api_key = APIKEY
        openai_api_base = "http://localhost:8000/v1"
        self.client = OpenAI(
            api_key=openai_api_key,
        )
        #self.offline_model=LLM(model="yentinglin/Llama-3-Taiwan-8B-Instruct-awq",quantization='AWQ', dtype="float16")
        self.model="gpt-4o-mini"
        self.prompt=PromptTemplate(prompt)
        self.temperature=temperature
        self.img_stm =img_memory
    def load_img(self,img_list=[],img_refresh=False)->list[dict]:
        send=[]
        # load img
        if self.img_stm != None:
            if img_list != []:
                if img_refresh:
                    self.img_stm.refresh()
                for img in img_list:
                    send.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img}",
                            "detail": "low"
                    }})
                    self.img_stm.save_img(img)
            else: 
                for img in self.img_stm.get_img():
                    if img==None:
                        break
                    send.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img}",
                            "detail": "low"
                        }})
        else:
            for img in img_list:
                send.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img}",
                        "detail": "low"
                }})
        return send
    def run_offline(self,text_dict: dict, img_list=[],img_refresh=False) -> str:
        send=self.load_img(img_list=[],img_refresh=False)
        text = self.prompt.format(text_dict)  
        print("==================================\n"+text+"\n====================================")
        send.append({
            "type": "text",
            "text": text
        })
        dev_prompt=self.prompt.get_dev_prompt()
        message = [
            {"role": "system","content": dev_prompt},
            {"role": "user","content": send}
            ]
        sampling_params = SamplingParams(temperature=0.8, top_p=0.65,max_tokens=1024)#max tokens要手動調整，default只有32
        outputs = self.offline_model.chat(message, sampling_params)

        generated_text = outputs[0].outputs[0].text
        return generated_text
        
    def run_online(self, text_dict: dict, img_list=[],img_refresh=False) -> str:
        send = self.load_img(img_list=[],img_refresh=False)
                    
        # load text 
        text = self.prompt.format(text_dict)  
        print("==================================\n"+text+"\n====================================")
        send.append({
            "type": "text",
            "text": text
        })

        # form request
        dev_prompt=self.prompt.get_dev_prompt()
        message = [
            {"role": "system","content": dev_prompt},
            {"role": "user","content": send}
            ]

        for i in range(5):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message,
                    temperature=self.temperature,
                    max_tokens=256
                    )
                output = str(response.choices[0].message.content)

                return output
            except Exception as e:
                print(e)
                continue
        return 'gpt error'