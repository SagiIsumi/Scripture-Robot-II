import requests

MODEL = "gpt-4o-mini"

class GPTopenai():
    def __init__(self, openai_api_key, prompt, temperature = 0.1, text_memory=None,model=MODEL, img_memory=None) -> None:
        self.key = openai_api_key
        self.model=model
        self.prompt = prompt
        self.text_stm = text_memory
        self.img_stm = img_memory
        self.temperature = temperature

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
        if self.text_stm != None:
            conversation = self.text_stm.get()
            text_dict['conversation']=conversation
        text = self.prompt.format(text_dict)
    
    
        print("==================================\n"+text+"\n====================================")
        send.append({
            "type": "text",
            "text": text
        })

        # form request
        message = [{
            "role": "user",
            "content": send
            }]
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.key}"
            }
        payload = {
            "model":self.model,
            "messages":  message,
            "temperature": self.temperature,
            "max_tokens": 1024
            }

        for i in range(5):
            try:
                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                j = response.json()
                output = str(j['choices'][0]['message']['content'])

                return j['choices'][0]['message']['content']
            except Exception as e:
                print(e)
                print(j["error"]["message"])
                continue
        return 'gpt error'
    