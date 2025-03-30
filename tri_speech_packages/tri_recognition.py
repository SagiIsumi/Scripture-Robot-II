import requests
import base64
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter() 
        result = func(*args, **kwargs)  
        end_time = time.perf_counter() 
        elapsed_time = end_time - start_time 
        print(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds")
        return result  
    return wrapper

# @timer
def recognize(file_path: str) -> str: # Implementing ASR in Multilingual using Liou's website api ASR
    
    # Read the audio file in binary mode
    with open(file_path, 'rb') as audio_file:
        base64_data = base64.b64encode(audio_file.read()).decode('utf-8')
    
    payload = {
        "data": [
            {"name": "audio.wav", "data": "data:audio/wav;base64," + base64_data}
        ]
    }

    url = "https://speech.bronci.com.tw/ai/taigi/run/predict"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        data = response.json() # {'data': [<text>], 'is_generating': False, 'duration': <time>, 'average_duration': <time>}
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return ""
    return data['data'][0]

if __name__ == "__main__":
    file_path = f"audio.wav"
    print(recognize(file_path))