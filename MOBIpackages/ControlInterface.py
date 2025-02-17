from MOBIpackages.trilingual_module import female_speak, minnan_speak2, normal_listen, minnan_listen
import cv2
import threading
import random
import pygame
import numpy as np
import os
from pygame.locals import KEYDOWN
import time
from TCPpackages.SocketClient import SocketClient
import pyaudio

class ControlInterface():
    def __init__(self, enable_camera=False, show_img=False, enable_arm=False, enable_face=False, is_FullScreen=False) -> None:
        self.enable_camera = enable_camera
        self.frame = None # 瞬時影像
        self.trigger=False
        self.face_emotion = 'neutral' # 機器人情緒
        self.action = 'nothing' # 機器人動作
        self.state = 'idol' # idol, speak, listen
        if enable_face:         # 臉部執行續
            self.t1 = threading.Thread(target=self.face_stream, args=(is_FullScreen,), daemon=False)
            self.t1.start()
            print('Face thread start')
        if enable_arm:         # 手臂執行續
            self.t2 = threading.Thread(target=self.arm_stream, daemon=True)
            self.t2.start()  
            print('Arm thread start')  
        if enable_camera:         # 相機執行續
            self.t3 = threading.Thread(target=self.camera_stream, args=(show_img,), daemon=True)
            self.t3.start()  
            print('Camera thread start')  

############################### 執行續定義 ###########################################
    def camera_stream(self, show_img) -> None: # 相機控制
        self.cap = cv2.VideoCapture(0)#我筆電先改成0
        if not self.cap.isOpened():
            print("Cannot open camera")
            return -1
        print('Camera start')
        try:
            while True:
                ret, self.frame = self.cap.read()
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                if show_img:
                    cv2.imshow("Frame", self.frame)
                cv2.waitKey(10)
                # 按下ctrl + q 跳出
                for event in pygame.event.get():
                    if event.type == KEYDOWN:
                        if event.key == pygame.K_q:
                            self.cap.release()
                            cv2.destroyAllWindows()
                            os._exit(1)
        except:
            self.cap.release()
            cv2.destroyAllWindows()
            return -1
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            return 0
    
    def face_stream(self, is_FullScreen=False) -> None: # 臉部控制
        valence = 0
        arousal = 0

        eyes = 0
        eyelids = 5
        eyelids2 = 5
        eye_lash = 0
        eye_size = 1

        mouth_v = 0
        mouth_size = 0

        pygame.init()
        if is_FullScreen:
            screen = pygame.display.set_mode((1024, 600), pygame.SCALED | pygame.RESIZABLE | pygame.FULLSCREEN, display=0)
        else:
            screen = pygame.display.set_mode((1024, 600))
        pygame.display.set_caption("PRESS ctrl + q TO EXIT")
        clock = pygame.time.Clock()

        va_flag = 1
        valence_new = valence
        arousal_new = arousal

        talk_s = 0.1
        mouth_talk = 0
        eye_talk = 0

        run = True
        pygame.mouse.set_visible(False)
        
        ball_radius = 10
        ball_color = (255, 255, 255)
        num_balls = 5
        jump_height = 50
        ball_spacing = 50  


        initial_x = 1024 // 2 - (ball_spacing * (num_balls - 1)) // 2
        initial_y = 600 // 2

        balls = [{'x': initial_x + i * ball_spacing, 'y': initial_y, 'direction': -1, 'max_height': initial_y - jump_height, 'jumping': False} for i in range(num_balls)]
        current_ball_index = 0
        while run:
            # Transferring state to parameters:
            if self.state == 'idol':
                is_listen = False
                is_speaking = False
            elif self.state == 'speak':
                is_listen = False
                is_speaking = True
            elif self.state == 'listen':
                is_listen = True
                is_speaking = False
            else:
                is_listen = False
                is_speaking = False    

            screen.fill(pygame.Color('black'))
            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    if event.key == pygame.K_q:
                        run = False
                        os._exit(1)
                    if event.key == pygame.K_l and is_listen == False:
                        is_listen = True
                    if event.key == pygame.K_n and (is_listen == True or is_speaking == True):
                        is_listen = False
                        is_speaking = False
                    if event.key == pygame.K_s and is_speaking == False:
                        is_speaking = True
                    # if event.key == pygame.K_s:
                    #     self.lock.acquire()
                    #     self.playing = False
                    #     self.lock.release()
                        
            if is_listen:
                for i, ball in enumerate(balls):
                    if i == current_ball_index:
                        pygame.draw.circle(screen, ball_color, (ball['x'], int(ball['y'])), ball_radius)
                        
                        if not ball['jumping']:
                            ball['jumping'] = True
                            ball['direction'] = -1
                        
                        if ball['jumping']:
                            ball['y'] += ball['direction'] * 5

                            if ball['y'] <= ball['max_height']:
                                ball['direction'] = 1
                            elif ball['y'] >= initial_y:
                                ball['direction'] = -1
                                ball['jumping'] = False
                                current_ball_index = (current_ball_index + 1) % num_balls  # 切换到下一个球
                    else:
                        pygame.draw.circle(screen, ball_color, (ball['x'], ball['y']), ball_radius)
                time.sleep(0.005)
                pygame.display.flip()
                
                continue
            current_ball_index = 0
            manual_robot_emotion = self.face_emotion
            if manual_robot_emotion == 'neutral':
                valence_new, arousal_new = 0.0, 0.0
            elif manual_robot_emotion == 'surprised':
                valence_new, arousal_new = np.cos(67.5 * np.pi / 180), np.sin(67.5 * np.pi / 180)
            elif manual_robot_emotion == 'excited':
                valence_new, arousal_new = np.cos(45 * np.pi / 180), np.sin(45 * np.pi / 180)
            elif manual_robot_emotion == 'happy':
                valence_new, arousal_new = np.cos(22.5 * np.pi / 180), np.sin(22.5 * np.pi / 180)
            elif manual_robot_emotion == 'pleased':
                valence_new, arousal_new = np.cos(-22.5 * np.pi / 180), np.sin(-22.5 * np.pi / 180)
            elif manual_robot_emotion == 'relaxed':
                valence_new, arousal_new = np.cos(-45 * np.pi / 180), np.sin(-45 * np.pi / 180)
            elif manual_robot_emotion == 'sleepy':
                valence_new, arousal_new = np.cos(-67.5 * np.pi / 180), np.sin(-67.5 * np.pi / 180)
            elif manual_robot_emotion == 'tired':
                valence_new, arousal_new = np.cos(-108 * np.pi / 180), np.sin(-108 * np.pi / 180)
            elif manual_robot_emotion == 'bored':
                valence_new, arousal_new = np.cos(-126 * np.pi / 180), np.sin(-126 * np.pi / 180)
            elif manual_robot_emotion == 'sad':
                valence_new, arousal_new = np.cos(-144 * np.pi / 180), np.sin(-144 * np.pi / 180)
            elif manual_robot_emotion == 'miserable':
                valence_new, arousal_new = np.cos(-162 * np.pi / 180), np.sin(-162 * np.pi / 180)
            elif manual_robot_emotion == 'disgust':
                valence_new, arousal_new = np.cos(157.5 * np.pi / 180), np.sin(157.5 * np.pi / 180)
            elif manual_robot_emotion == 'angry':
                valence_new, arousal_new = np.cos(135 * np.pi / 180), np.sin(135 * np.pi / 180)
            elif manual_robot_emotion == 'fear':
                valence_new, arousal_new = np.cos(112.5 * np.pi / 180), np.sin(112.5 * np.pi / 180)
            else:
                valence_new, arousal_new = 0.3, 0.0

            if ((abs(valence_new - valence) < 0.001) and (abs(arousal_new - arousal) < 0.001)) or (va_flag == 1) or (Valence_new != valence_new) or (Arousal_new != arousal_new):
                Valence_new = valence_new
                Arousal_new = arousal_new
                dv = (Valence_new - valence) / 10
                da = (Arousal_new - arousal) / 10
                va_flag = 0

            valence = valence + dv
            mouth_v = mouth_v + dv
            
            arousal = arousal + da
            eye_lash = eye_lash + da
            eye_size = eye_size + da
            mouth_size = mouth_size + da
        
            if eyes < -30: 
                eyes = -30

            if eyes > 30: 
                eyes = 30


            if is_speaking:
                clock.tick(20)
                if mouth_size<=0:
                    if mouth_size+mouth_talk >= 0.3:
                        talk_s = -0.03
                    if mouth_size+mouth_talk <= 0:
                        talk_s = 0.03
                else:
                    if mouth_size + mouth_talk >= mouth_size + 0.2:
                        talk_s = -0.03
                    if mouth_size + mouth_talk <= mouth_size - 0.2:
                        talk_s = 0.03
                mouth_talk = mouth_talk + talk_s
                eye_talk = eye_talk + talk_s
            else:
                clock.tick(30)
                if (mouth_talk < 0.001 and mouth_talk > -0.001) or (eye_talk < 0.001 and eye_talk > -0.001):
                    pass
                if mouth_talk > 0:
                    mouth_talk = mouth_talk - 0.1
                if mouth_talk < 0:
                    mouth_talk = mouth_talk + 0.1
                if eye_talk > 0:
                    eye_talk = eye_talk - 0.1
                if eye_talk < 0:
                    eye_talk = eye_talk + 0.1 
                
            # plot eyes
            pygame.draw.ellipse(screen, [127, 255, 212], (100 - eyelids2 * 0.2, 100, 280 + eyes, (280 + eyes - eyelids * 2.2) * (((eye_size + eye_talk) / 10) + 1) ), 30)
            pygame.draw.ellipse(screen, [127, 255, 212], (670 + eyelids2 * 0.2, 100, 280 + eyes, (280 + eyes - eyelids * 2.2) * (((eye_size + eye_talk) / 10) + 1) ), 30)

            # plot blush
            pygame.draw.ellipse(screen, [251, 147, 194], (50, 410, 100, 40), 0)
            pygame.draw.ellipse(screen, [251, 147, 194], (875, 410, 100, 40), 0)  

            mouth_width = 100
            mouth_height = 550

            # plot mouth
            for i in range(0, mouth_width):                                   
                x = i
                y = -0.006 * mouth_v * (x - 3) * (x - 1) + valence * 20  # sine wave formula
                
                pygame.draw.ellipse(screen, [127, 255, 212], [x + 520, y + mouth_height, 10, 10], 0)
                if mouth_size+mouth_talk >= 0:
                    pygame.draw.ellipse(screen, [127, 255, 212], [x + 520, y + mouth_height - 1.5 * ((((valence + 2) / 1.5) + 1) * 0.3) * (mouth_size + mouth_talk) * (mouth_width - x), 10, 10], 0)

            for i in range(0, -mouth_width, -1):                                   
                x = i
                y = -0.006 * mouth_v * (x + 2) * (x - 2) + valence * 20  # sine wave formula

                pygame.draw.ellipse(screen, [127, 255, 212], [x + 520, y + mouth_height, 10, 10], 0)
                if mouth_size + mouth_talk >= 0:
                    pygame.draw.ellipse(screen, [127, 255, 212], [x + 520, y + mouth_height - 1.5 * ((((valence + 2) / 1.5) + 1) * 0.3) * (mouth_size + mouth_talk) * (mouth_width + x), 10, 10], 0)

            # plot eyebrow
            pygame.draw.lines(screen, [127, 255, 212], 0, [(120, -arousal * 50 + 80), (340, -arousal * 20 + 80 + eye_lash * 70)], 20)
            pygame.draw.lines(screen, [127, 255, 212], 0, [(710, -arousal * 20 + 80 + eye_lash * 70), (930, -arousal * 50 + 80)], 20)

            # plot frown (left)
            s = pygame.Surface((20, 40 + arousal * 20))
            s.set_alpha(0 - valence * 255)
            s.fill([127, 255, 212])
            screen.blit(s, (321, -arousal * 20 + 80 + eye_lash * 70 - (40 + arousal * 20)))    # (0,0) are the top-left coordinates
            
            # plot frown (right)
            s = pygame.Surface((20, 40 + arousal * 20))
            s.set_alpha(0 - valence * 255)
            s.fill([127, 255, 212])
            screen.blit(s, (710, -arousal * 20 + 80 + eye_lash * 70 - (40 + arousal * 20)))    # (0,0) are the top-left coordinates
            
            pygame.display.flip()

    def arm_stream(self) -> None: # 手臂控制
        arm_send = SocketClient('127.0.0.1', 12345)
        arm_recv = SocketClient('127.0.0.1', 4478)
        while True:
            if self.action != 'nothing':
                arm_send.send_msg(self.action)
                recv = arm_recv.wait_msg()
                if recv != 'ok':
                    print('Error: Action failed')
                self.action = 'nothing'
                continue
            else:
                continue

############################### 調用函數 ###########################################
    def get_frame(self) -> None: # 取得影像
        cv2.imwrite('./input_img/img.jpg', self.frame)
        print('save')

    def random_action(self) -> str:  # 隨機動作選擇
        index = random.randint(1, 6)
        if index == 1:
            return 'rnd1'
        elif index == 2:
            return 'rnd2'
        elif index == 3:
            return 'rnd3'
        elif index == 4:
            return 'rnd4'
        else:
            return 'nothing'
        
############################### 外部調用函數 ###########################################    
    def wait_input(self, language='ch') -> str: # 接聽、捕捉影像
        while True:
            self.state = 'listen'
            # 儲存影像
            if self.enable_camera:
                self.get_frame()
            # 語音辨識
            if language == 'ch':
                text = normal_listen()
            elif  language == 'eng':
                text = normal_listen()
            elif  language == 'minnan':
                text = minnan_listen()
            else:
                text = normal_listen()

            if text.find('無法辨識您的語音') != -1:
                self.state = 'speak'
                self.express(self,'對不起，我沒聽清楚，請再說一遍好嗎?', 'sad', 'notihng',  language=language)
                continue
            else:
                self.state = 'idol'
                return text
    def inner_female_speak(self,input_text):
        female_speak(input_text,volume=1,speed='fast',tone='normal')
        self.trigger=True
    def inner_minnan_speak(self,input_text):
        minnan_speak2(input_text)
        self.trigger=True       
    def express(self, talk, emotion, action, language='ch') -> None: # 表達  
        # arm
        self.action = action 
        if action == 'nothing':
            self.action = self.random_action() 
        # face
        self.face_emotion = emotion 
        self.state = 'speak'
        # voice
        frames=[]
        audio_format=pyaudio.paInt16
        channels = 1
        rate = 44100
        chunk = 1024
        interrupt=False
        self.trigger=False
        try:
            if language == 'ch' or language=='en':
                speaker=threading.Thread(target=self.inner_female_speak, args=(talk,), daemon=True)
                speaker.start()#播音執行緒
            else:
                speaker=threading.Thread(target=self.inner_minnan_speak, args=(talk,), daemon=True)
                speaker.start()#播音執行緒
        except Exception as e:
            print(e)
        p=pyaudio.PyAudio()
        detecting_threashold=55#音量閾值
        stream=p.open(format=audio_format,
                channels=channels,
                rate=rate,
                input=True,
                #input_device_index=2,
                frames_per_buffer=chunk)
        try:
            while True:#開始收音
                for i in range(12):
                    data = stream.read(chunk)
                    frames.append(data)
                audio_data = np.frombuffer(b"".join(frames), dtype=np.int16)
                volume = np.abs(audio_data).mean()
                print(volume)
                if volume>detecting_threashold:#音量大於閾值結束播音
                    pygame.mixer.music.stop()
                    interrupt=True
                    print("over")
                    break
                elif self.trigger:
                    interrupt=False
                    break
        except Exception as e:
            print(e)
        stream.stop_stream()
        stream.close()
        p.terminate()#關閉串流
        self.state = 'idol'
        return interrupt
    

if __name__ == '__main__':
    interface = ControlInterface(enable_camera=False, show_img=False, enable_arm=True, enable_face=True, is_FullScreen=False)
    interface.express('大家好，我叫做莫比', 'happy', 'test', language='minnan')
    print('end')
    exit()

