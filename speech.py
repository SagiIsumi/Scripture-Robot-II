from pathlib import Path
from datetime import datetime
import requests
import pyaudio
import wave
import threading
import pygame
from datetime import datetime
from test_liou_api import recognize
import numpy as np
from trilingual_module import female_speak,minnan_speak2
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

    def inner_female_speak(self,input_text):
        female_speak(input_text,volume=1,speed='fast',tone='normal')
        self.triger=True
    def inner_minnan_speak(self,input_text):
        minnan_speak2(input_text)
        self.triger=True 
    def speaking(self,text,language='ch')->None:#播音
        frames=[]
        interrupt=False
        self.triger=False
        try:
            if language == 'chinese' or language=='english':
                speaker=threading.Thread(target=self.inner_female_speak, args=(text,), daemon=True)
                speaker.start()#播音執行緒
            else:
                speaker=threading.Thread(target=self.inner_minnan_speak, args=(text,), daemon=True)
                speaker.start()#播音執行緒
        except Exception as e:
            print(e)
        p=pyaudio.PyAudio()
        detecting_threashold=55#音量閾值
        stream=p.open(format=self.audio_format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                #input_device_index=2,
                frames_per_buffer=self.chunk)
        try:
            while True:#開始收音
                for i in range(12):
                    data = stream.read(self.chunk)
                    frames.append(data)
                audio_data = np.frombuffer(b"".join(frames), dtype=np.int16)
                volume = np.abs(audio_data).mean()
                print(volume)
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
        threashold=60 #音量閾值
        max_volume_threashold=45
        silent_chunk=0 #沉默時長
        silent_duration=3
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
        if max_volume<max_volume_threashold:#若收音為無聲音檔返回None
            return "None"

        time_object=datetime.now()
        currentTime = time_object.strftime("%d-%m-%y_%H-%M-%S")
        audio_path=Path("./audio_file")/Path(currentTime+".wav")
        Path(audio_path).parent.mkdir(parents=True, exist_ok=True)
        audio_path=str(audio_path)
        with wave.open(audio_path,"wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(p.get_sample_size(self.audio_format))
            wf.setframerate(self.rate)
            wf.writeframes(b"".join(frames))#音檔寫出
        return audio_path#返回檔案儲存路徑
    
    def check_language(self,path)->str:
        client=OpenAI(api_key=openai_key)
        audio=open(path,"rb")
        response=client.audio.transcriptions.create(
            model="whisper-1",
            file=audio,
            response_format="verbose_json")
        print(response)
        response=client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":f"你覺得這是正統中文、英文還是台灣方言?\
                       若是正統中文回答:chinese，若是英文回答:english，若是台灣方言回答:taigi。提問:{response.text}"}]
        )
        lg=response.choices[0].message.content
        lg=re.search('(chinese)|(taigi)|(english)',lg,re.I).group()
        return lg
            
    def speech_to_text(self,path)->str:
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