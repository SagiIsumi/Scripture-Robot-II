from pathlib import Path
from datetime import datetime
import requests
import pyaudio
import wave
import threading
import pygame
from datetime import datetime
from tri_speech_packages.tri_recognition import recognize
import numpy as np
from tri_speech_packages.trilingual_module import female_speak,minnan_speak2
from openai import OpenAI
import configparser
from opencc import OpenCC
import re

config=configparser.ConfigParser()
config.read('config.ini')
openai_key=config.get('openai','key1')
converter = OpenCC('s2t.json')


class audio_procession(): 
    def __init__(self) -> None:
        self.audio_format=pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.chunk = 1024
        self.triger=False

    def inner_female_speak(self,input_text): #內部播音執行緒，多加了trigger去控制外部聲音介入
        female_speak(input_text,volume=1,speed='normal',tone='normal')
        self.triger=True
    def inner_minnan_speak(self,input_text): #內部播音執行緒，多加了trigger去控制外部聲音介入
        minnan_speak2(input_text)
        self.triger=True 
    def speaking(self,text,language='ch')->None:#播音
        frames=[]
        interrupt=False
        self.triger=False
        try: #採用thread的方式同時播音並且監聽外部聲音確認是否中斷對話
            if language == 'chinese' or language=='english':
                speaker=threading.Thread(target=self.inner_female_speak, args=(text,), daemon=True)
                speaker.start()#播音執行緒
            else:
                speaker=threading.Thread(target=self.inner_minnan_speak, args=(text,), daemon=True)
                speaker.start()#播音執行緒
        except Exception as e:
            print(e)
        p=pyaudio.PyAudio()
        detecting_threashold=24#音量閾值，自己設定，每台電腦的靈敏度不一樣
        stream=p.open(format=self.audio_format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                #input_device_index=2,
                frames_per_buffer=self.chunk)
        while pygame.mixer.get_init()==None: #播音緒執行後才開啟監聽緒
            continue
        try:
            while True:#開始監聽
                for i in range(12):
                    data = stream.read(self.chunk)
                    frames.append(data)
                audio_data = np.frombuffer(b"".join(frames), dtype=np.int16)
                volume = np.abs(audio_data).mean()
                #print(volume)
                if volume>detecting_threashold:#音量大於閾值結束播音
                    pygame.mixer.music.stop()
                    interrupt=True
                    print("over")
                    break
                elif self.triger:
                    interrupt=False
                    break
        except Exception as e:
            print(e)
        stream.stop_stream()
        stream.close()
        p.terminate()#關閉串流
        return interrupt
            
    def recording(self)->str:
        p=pyaudio.PyAudio()
        frames=[]
        threashold=100 #音量閾值，自己設定，每台電腦的靈敏度不一樣
        max_volume_threashold=95#平均音量閾值，若小於此值則視為無聲音檔
        silent_chunk=0 #沉默時長的count
        silent_duration=3 #沉默時長，簡單說要音量閾值都大於一定沉默時長才是為正常對話，否正視為環境噪音
        silent_chunks_threshold = int(silent_duration*self.rate/self.chunk)
        try:
            stream=p.open(format=self.audio_format,
                    channels=self.channels,
                    rate=self.rate,
                    input=True,
                    #input_device_index=0,
                    frames_per_buffer=self.chunk)
            print("開始錄音")
        except Exception as e:
            print(e)
        try:#持續收音直到沉默時長<沉默閾值
            while True:
                data = stream.read(self.chunk)
                frames.append(data)
                audio_data = np.frombuffer(data, dtype=np.int16)
                volume = np.abs(audio_data).mean()
                if volume < threashold:
                    silent_chunk+=1
                else:
                    silent_chunk=0
                if silent_chunk>=silent_chunks_threshold:
                    break
        except Exception as e:
            print(e)
        stream.stop_stream()
        stream.close()
        p.terminate()#關閉串流
        #檔案輸出_wav檔

        verify_data=np.frombuffer(b"".join(frames), dtype=np.int16)
        max_volume = np.abs(verify_data).mean()
        print(max_volume)
        if max_volume<max_volume_threashold:#若收音為無聲音檔返回None
            return "None"

        time_object=datetime.now()
        currentTime = time_object.strftime("%d-%m-%y_%H-%M-%S")
        audio_path=Path("./audio_file")/Path(currentTime+".wav")#紀錄時間
        Path(audio_path).parent.mkdir(parents=True, exist_ok=True)#創建路徑
        audio_path=str(audio_path)
        with wave.open(audio_path,"wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(p.get_sample_size(self.audio_format))
            wf.setframerate(self.rate)
            wf.writeframes(b"".join(frames))#音檔寫出
        return audio_path#返回檔案儲存路徑
    
    def check_language(self,path)->str: #語言判斷，但表現有不穩定性，未來我考慮和打斷功能合併並一起訓練個模型去處理
        client=OpenAI(api_key=openai_key)
        audio=open(path,"rb")
        response=client.audio.transcriptions.create(
            model="whisper-1",
            file=audio,
            response_format="verbose_json")#Speech To Text
        response=client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "developer","content": "你是負責判讀語言種類的助理，請根據提問，回答提問所使用的語言是下列三者中的哪一個。chinese、english、taigi，請從中三選一。"},
                {"role":"user","content":f"你覺得這是正統中文、英文還是台灣方言?\
                       若是正統中文回答:chinese，若是英文回答:english，若是台灣方言回答:taigi。提問:{response.text}"}]
        )#讓GPT判斷語言，但目前表現不穩定
        lg=response.choices[0].message.content
        try:
            lg=re.search('(chinese)|(taigi)|(english)',lg,re.I).group()
        except Exception as e:
            print(lg)
            print(e)
        return lg
            
    def speech_to_text(self,path)->str:#三語言的語音辨識，這邊是用Liou的網站api
        return recognize(file_path=path)

        
def Main(): #測試用
    test=audio_procession()
    results=test.check_language("audio.wav")
    print(results)

if __name__=='__main__':
    Main()

# time_object=datetime.now()
# currentTime = time_object.strftime("%d-%m-%y_%H-%M-%S")
# print(currentTime)
# audio_path=Path("./audio_file")/Path(currentTime+".mp3")
# print(audio_path)