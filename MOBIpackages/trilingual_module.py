import pygame
import threading
import speech_recognition as sr
from gtts import gTTS
# from datetime import datetime
import librosa
import pyttsx3
import soundfile as sf
import time
import requests
# 可選擇的mp3採樣率為8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000

# T2S
def play_mp3(FilePath, volume = 0.5):
    pygame.mixer.init()
    pygame.mixer.music.load(FilePath)
    pygame.mixer.music.set_volume(volume)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy() == True:
        continue
    pygame.mixer.music.stop()
    pygame.mixer.music.unload()

def female_speak(input_text, volume=0.5, speed='normal', tone='normal'):
    folder = r"voice//"
    localtime = time.localtime()
    filename = folder + str(time.strftime("%Y-%m-%d-%I-%M-%S-%p", localtime)) + '.mp3'
    tts = gTTS(text=input_text, lang='zh-TW')
    tts.save(filename)
    # play_mp3(filename)
    if speed=='faster':
        y,sr = librosa.load(filename)
        b = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=-7)
        sf.write(filename, b, 32000)
    elif speed=='fastest':
        y,sr = librosa.load(filename)
        b = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=-12)
        sf.write(filename, b, 44100)
    elif speed == 'slower':
        y,sr = librosa.load(filename)
        b = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=6)
        sf.write(filename, b, 16000)
    
    if tone == 'normal':
        play_mp3(filename, volume)
    elif tone == 'higher':
        y,sr = librosa.load(filename)
        b = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=0.8)
        sf.write(filename, b, 24000)
        play_mp3(filename, volume)
    elif tone == 'lower':
        y,sr = librosa.load(filename)
        b = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=-5)
        sf.write(filename, b, 24000)
        play_mp3(filename, volume)
    elif tone == 'lowest':
        y,sr = librosa.load(filename)
        b = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=-7)
        sf.write(filename, b, 24000)
        play_mp3(filename, volume)

def male_speak(input_text, volume=1.0, speed='normal', tone='normal'):
    folder = r"voice//"
    localtime = time.localtime()
    filename = folder + str(time.strftime("%Y-%m-%d-%I-%M-%S-%p", localtime)) + '.mp3'
    engine = pyttsx3.init()
    # engine.setProperty('volume', volume)

    if speed=='faster':
        rate = engine.getProperty('rate')
        engine.setProperty('rate', rate+70)
    elif speed=='fastest':
        rate = engine.getProperty('rate')
        engine.setProperty('rate', rate+140)
    elif speed == 'slower':
        rate = engine.getProperty('rate')
        engine.setProperty('rate', rate-70)

    if tone == 'normal':
        engine.setProperty('volume', volume)
        engine.say(input_text)
        engine.runAndWait()
    elif tone == 'higher':
        engine.save_to_file(input_text, filename)
        engine.runAndWait()
        y,sr = librosa.load(filename)
        b = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=1)
        sf.write(filename, b, 24000)
        play_mp3(filename, volume)
    elif tone == 'lower':
        engine.save_to_file(input_text, filename)
        engine.runAndWait()
        y,sr = librosa.load(filename)
        b = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=-4)
        sf.write(filename, b, 24000)
        play_mp3(filename, volume)
    elif tone == 'lowest':
        engine.save_to_file(input_text, filename)
        engine.runAndWait()
        y,sr = librosa.load(filename)
        b = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=-7)
        sf.write(filename, b, 24000)
        play_mp3(filename, volume)

def minnan_speak2(sentence, accent=0, gender=1, volume=1.0):
    accent_args = ['強勢腔（高雄腔）','次強勢腔（台北腔）']
    gender_args = ['男聲','女聲']
    text1 = requests.get(f'http://tts001.iptcloud.net:8804/display?text0={sentence}').content.decode() # translate to TLPA (Taiwan Language Phonetic Alphabet)
    r = requests.get(f'http://tts001.iptcloud.net:8804/synthesize_TLPA?text1={text1}&gender={gender_args[gender]}&accent={accent_args[accent]}')
    with open('./voice/output.wav', 'wb') as file:
        file.write(r.content)
    file.close()
    play_mp3('./voice/output.wav', volume=volume)

# S2T
def normal_listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source,duration=1)
        print("請開始說話...")
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio, language='zh-TW')
        # print("您說的是：" + text)
    except sr.UnknownValueError:
        # print("無法辨識您的語音")
        text = "無法辨識您的語音"
    except sr.RequestError as e:
        # print("無法連線至Google語音辨識服務：{0}".format(e))
        text = "無法連線至Google語音辨識服務：{0}".format(e)

    return text

def minnan_listen():
    r = sr.Recognizer()
    with sr.Microphone(sample_rate=16000) as source:
        r.adjust_for_ambient_noise(source,duration=1)
        print("請開始說話...")
        audio = r.listen(source)
    localtime = time.localtime()
    url = "http://119.3.22.24:3998/dotcasr"
    folder = "./voice"
    filename = folder + '/' + str(time.strftime("%Y-%m-%d-%I-%M-%S-%p", localtime)) + '.wav'
    with open(filename, 'wb') as file:
        file.write(audio.get_wav_data())

    with open(filename, 'rb') as file:
        data = {'userid': '00001 ', 'token': '123356'}
        response = requests.post(url, data=data, files={"file": file})
        j = response.json()
    # print(response.json())
    return j['result']


if __name__ == '__main__':
   female_speak('你好，我是莫比', volume=1.0, speed='faster', tone='normal')
