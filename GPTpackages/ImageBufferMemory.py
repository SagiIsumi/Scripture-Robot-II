class ImageBufferMemory():
    def __init__(self) -> None:
        self.buffer = []

    def refresh(self)-> None:
        self.buffer = []

    def save_img(self, img) -> None:
        self.buffer.append(img)

    def get_img(self) -> list:
        return self.buffer

import base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')