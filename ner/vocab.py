import json
import os
from loggers import logger

class Vocab(object):
    """
    构建词或者字的id映射
    """
    def __init__(self,vocab2id=None,id2vocab=None,tag2id=None,id2tag=None,unk_vocab_id=0,pad_vocab_id=1,pad_tag_id=0):
        self.vocab2id=vocab2id
        self.id2vocab=id2vocab
        self.tag2id=tag2id
        self.id2tag=id2tag
        self.unk_vocab_id=unk_vocab_id
        self.pad_vocab_id=pad_vocab_id
        self.pad_tag_id=pad_tag_id
        self.vocab_size = 0 if not vocab2id else len(vocab2id)
        self.label_size = 0 if not tag2id else len(tag2id)

    def build_vocab(self,texts=None,labels=None,build_texts=True,build_labels=True,with_build_in_tag_id=True):
        logger.info("Start building vocabulary,please wait for a while......")
        if build_texts:
            #初始化一个字典
            assert texts,"请确保texts不为空"
            self.vocab2id={"<UNK>":self.unk_vocab_id,"<PAD>":self.pad_vocab_id}
            total_index=2
            for text in texts:
                for seg in text:
                    if seg in self.vocab2id:
                        continue
                    self.vocab2id[seg]=total_index
                    total_index+=1
            self.id2vocab={value:key for key,value in self.vocab2id.items()}
            self.vocab_size=len(self.vocab2id)

        if build_labels:
            assert labels,"请确保labels不为空"
            self.tag2id={"<PAD>":self.pad_tag_id}
            total_index=1
            if not  with_build_in_tag_id:
                self.tag2id={}
                total_index=0
            for label in labels:
                for each_label in label:
                    if each_label in self.tag2id:
                        continue
                    self.tag2id[each_label]=total_index
                    total_index+=1
        self.label_size=len(self.tag2id)
        self.id2tag={value:key for key,value in self.tag2id.items()}
        logger.info("Build vocabulary finished,vocab_sie:{},label_size:{} .".format(self.vocab_size,self.label_size))

    def save_vocab(self,vocab_file):
        result={
            "vocab2id":self.vocab2id,
            "id2vocab":self.id2vocab,
            "tag2id":self.tag2id,
            "id2tag":self.id2tag
        }
        #将上述字典以json的格式写入指定路径
        try:
            with open(vocab_file,"w",encoding="utf-8") as doc:
                doc.write(json.dumps(result,ensure_ascii=False, indent=4))
        except Exception as err:
            print(err)
        logger.info("save vocab to {}".format(vocab_file))

    def loadvocab(self,vocab_file):
        with open(vocab_file,"r",encoding="utf-8") as doc:
            result=json.loads(doc.read())
            self.vocab2id=result["vocab2id"]
            self.id2vocab=result["id2vocab"]
            self.tag2id=result["tag2id"]
            self.id2tag=result["id2tag"]
            self.tagsize=len(self.tag2id)
            self.vocabsize=len(self.vocab2id)

    """
    当使用预训练的模型时，可能会使用外部的vocab，下面的方法允许设置外部的vocab
    """

    def set_vocab2id(self, vocab2id):
        self.vocab2id = vocab2id
        self.vocab_size = len(self.vocab2id)
        return self

    def set_id2vocab(self, id2vocab):
        self.id2vocab = id2vocab
        return self

    def set_tag2id(self, tag2id):
        self.tag2id = tag2id
        self.label_size = len(self.tag2id)
        return self

    def set_id2tag(self, id2tag):
        self.id2tag = id2tag
        return self

    def set_unk_vocab_id(self, unk_vocab_id):
        self.unk_vocab_id = unk_vocab_id

    def set_pad_vocab_id(self, pad_vocab_id):
        self.pad_vocab_id = pad_vocab_id

    def set_pad_tag_id(self, pad_tag_id):
        self.pad_tag_id = pad_tag_id