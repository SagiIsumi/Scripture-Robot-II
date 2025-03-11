class TextBuffer():
    def __init__(self, buffer_size=5)->None:
        self.buffer_size = buffer_size
        self.buffer = []
    
    def set(self, con:str) -> None:
        self.buffer.append(con)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def get(self,require_num=5) ->str:
        text = ''
        for i in self.buffer[0:require_num]:
            text=text+i+"\n"
        return text
