from ahocorasick import Automaton
from ner import NER_TYPES

class NerDict(object):

    def __init__(self,dict_path=None,splitter="\t"):
        self.splitter=splitter
        self.autos={ner_type:None for ner_type in NER_TYPES} #每一个类型对应一个自动机
        self.words={ner_type: set() for ner_type in NER_TYPES} #{"ORG":(),"PER":{},"LOC":(),"TIME":()}
        if dict_path:
            self._load(dict_path)

    def _load(self,dict_path):
        with open(dict_path,'r',encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line or line.startswith("#"):
                    continue
                word, ner_type=line.split(self.splitter)[:2]
                self.words[ner_type].add(word)  #集合添加元素

        #根据加载好的词典初始化自动机
        for ner_type in NER_TYPES:
            if not self.words[ner_type]:
                continue
            self.autos[ner_type]=self._build_auto(self.words[ner_type])

    @staticmethod
    def _build_auto(words):
        auto=Automaton()
        [auto.add_word(word,word) for word in words]
        auto.make_automaton()
        return auto

    def findall(self,content,types=NER_TYPES,with_offset=True):
        """
        从文本中查找实体词
        """
        result={}
        for ner_type in types:
            if not self.autos[ner_type]:
                continue
            result[ner_type]=[]
            for end_idx,word in self.autos[ner_type].iter(content):
                result[ner_type].append((end_idx+1-len(word),word) if with_offset else word)

            if not result[ner_type]:
                del result[ner_type]
        return result

    def add_word(self,word,ner_type):
        self.words[ner_type].add(word)
        self.autos[ner_type]=self._build_auto(self.words[ner_type]) #每个类型每添加一个词，则对应类型的自动机就会重新构建一次
        pass

    def delete_word(self,word,ner_type):
        self.words[ner_type].remove(word)
        if self.words[ner_type]:
            self.autos[ner_type]=self._build_auto(self.words[ner_type])
        else:
            self.autos[ner_type]=None