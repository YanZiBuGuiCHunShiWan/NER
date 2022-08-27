import unittest
from utils import NerDict

class Mytest(unittest.TestCase):

    def setUp(self) -> None:
        self._dict_path= "../utils/rule_dict/dict.txt"
        self.ner_dict=NerDict(dict_path=self._dict_path,splitter=" ") #加载词典,并构建AC自动机

    def test1(self):
        text="有一天，张三李四和王五准备出去玩，但是路过珠海市第二中学的时候停下了步伐"
        print("测试结果:",self.ner_dict.findall(content=text))


    def test2(self):
        #测试添加和删除词语的功能
        self.ner_dict.add_word("广东省珠海市","LOC")
        self.ner_dict.add_word("广西壮族自治区","LOC")
        self.ner_dict.delete_word("李四", "PER")
        text="有一天，张三李四和王五准备出去玩，但是路过珠海市第二中学的时候停下了步伐,随后几人心想去广西壮族自治区玩"
        print("测试结果:",self.ner_dict.findall(text))