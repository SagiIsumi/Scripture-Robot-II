import re
class PromptTemplate():
    def __init__(self, path:str) -> None:
        with open(path, 'r', encoding='utf-8') as f:
            self.p = f.read()
        self.pattern1=re.compile("{.+}",re.U)
        self.pattern2=re.compile("(\s|.)+?\n\n",re.U)
        #print(self.prompt_chuncks)
    def get_dev_prompt(self):
        prompt=self.pattern2.search(self.p).group()
        return prompt
    def format(self, input_dict:dict)->str:
        true_keylist=input_dict.keys()
        new_txt=re.search('\n\n(\s|.)+',self.p,re.U).group()
        for pattern in self.pattern1.finditer(new_txt):
            key=re.search("(\w|[\u4e00-\u9fff])+",pattern.group(),re.U).group()
            if key in true_keylist:
                new_txt=re.sub(pattern.group(),input_dict[key],new_txt,re.U)
        return new_txt
