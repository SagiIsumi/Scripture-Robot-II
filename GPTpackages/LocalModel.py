from openai import OpenAI
from GPTpackages.PromptTemplate import PromptTemplate

class local_LLM():
    def __init__(self,prompt:str,model:str="yentinglin/Llama-3-Taiwan-8B-Instruct-awq",
                 temperature:int=1, img_memory=None):
        openai_api_key = "EMPTY"
        openai_api_base = "http://140.112.14.207:8000/v1"
        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        self.model=model
        self.prompt=PromptTemplate(prompt)
        self.temperature=temperature
        self.img_stm =img_memory

    def run(self, text_dict: dict, img_list=[],img_refresh=False) -> str:
        send = []
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
            {"role": "developer","content": dev_prompt},
            {"role": "user","content": send}
            ]

        for i in range(5):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message,
                    temperature=self.temperature,
                    )
                output = str(response.choices[0].message.content)

                return output
            except Exception as e:
                print(e)
                continue
        return 'gpt error'