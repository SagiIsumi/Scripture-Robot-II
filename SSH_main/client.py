from argparse import ArgumentParser
from GPTpackages.ImageBufferMemory import encode_image
from tri_speech_packages.trilingual_module import female_speak, minnan_speak2
from MOBIpackages import ControlInterface
from tri_speech_packages.speech import audio_procession
from TCPpackages import GPU_Client
from pathlib import Path
import asyncio
import threading
import time

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

#以下是啟用各項物件
MyAudio=audio_procession()
interface=ControlInterface.ControlInterface(enable_camera=False, show_img=False, enable_arm=True, enable_face=True, is_FullScreen=True)
client = SocketClient('140.112.14.248', 12345)


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
    
    #主程式
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
            language, text, emotion  = asyncio.run(input_trans(MyAudio,user_input))
        else:
            start_time=time.perf_counter()
            text = input("請輸入:")
            language = 'chinese'
            emotion = 'neutral'
            
        #legacy code    
        """
        #interface.get_frame()
        #fileList = os.listdir('input_img')
        # fileList = []
        # if fileList != []:
        #     img_list = [encode_image('input_img/' + fileList[-1])]
        """
            
        print("text: ",text)
        print("language: ", language)
        if text =="對話結束" or text =="對話結束。":
            break
        text_dict['what']=text
        text_dict['language']=language
        end_time1=time.perf_counter()
        query=f"what:{text}, language:{language}"
        #interrupt=MyAudio.speaking(result,language=language) #單獨播音(debug)
        
        #SSH client
        client.send_msg()
        result = client.wait_msg()

        interrupt=interface.express(result, emotion, 'nothing', language=language)#整合模型
        end_time2=time.perf_counter()
        elapsed_time1=end_time1-start_time
        elapsed_time2=end_time2-start_time
        print(emotion)
        print("generate_time",elapsed_time1)
        print("totaltime:", elapsed_time2)
    print('END!')
