class TextBuffer():
    def __init__(self, buffer_size=1)->None:
        self.buffer_size = buffer_size
        self.buffer = []
    
    def set(self, con:list) -> None:
        self.buffer.append(con)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def get(self,require_num=-1) ->str:
        text = ''
        for i,b in enumerate(self.buffer):
            if require_num!=-1:
                if i <(len(self.buffer)-require_num-1):
                    continue
            text = text + "Human:" +b[0]+'\n'+"莫比:"+b[1]+'\n'

        return text
