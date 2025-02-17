class PromptTemplate():
    def __init__(self, path:str) -> None:
        with open(path, 'r', encoding='utf-8') as f:
            p = f.read()
        chunks = str(p).split('{')[1:]
        self.variables = []
        for chunk in chunks:
            variable = chunk.split('}')[0]
            self.variables.append(variable)
            p = p.replace('{' + variable + '}',r'{}')
        self.prompt_chuncks = p.split(r'{}')
        #print(self.prompt_chuncks)
        
    def format(self, input_dict:dict)->str:
        KeyList = list(input_dict.keys())
        index = 0
        prompt = self.prompt_chuncks[index]
        for variable in self.variables:
            index = index + 1
            if variable not in KeyList:
                prompt = prompt + '' + self.prompt_chuncks[index] 
            else:
                prompt = prompt + str(input_dict[variable]) + self.prompt_chuncks[index] 
        return prompt