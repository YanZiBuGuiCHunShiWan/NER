import torch.nn as nn
from transformers import AlbertTokenizer, AlbertModel
from transformers import BertTokenizer, BertModel
from transformers import ElectraTokenizer, ElectraModel

from ner.crf.crf_layer import CRF

class BertCRF(nn.Module):
    """
    大致功能是，要把BERT的输出的Logits传给Crflayer去解码
    """
    def __init__(self,bert_model_dir,label_size,dropout_rate=0.3):
        super(BertCRF,self).__init__()
        self.label_size=label_size

        #根据不同路径选择初始化不同对象
        if "albert" in bert_model_dir.lower():
            self.bert_tokenizer=BertTokenizer.from_pretrained(bert_model_dir)
            self.bert_model=AlbertModel.from_pretrained(bert_model_dir)

        elif "electra" in bert_model_dir.lower():
            self.bert_tokenizer=BertTokenizer.from_pretrained(bert_model_dir)
            self.bert_model=ElectraModel.from_pretrained(bert_model_dir)

        else:
            self.bert_tokenizer=ElectraTokenizer.from_pretrained(bert_model_dir)
            self.bert_model=BertModel.from_pretrained(bert_model_dir)

        self.dropout=nn.Dropout(dropout_rate)
        self.linear=nn.Linear(self.bert_model.config.hidden_size,label_size)  #将BERT输出的embedd_dim 映射到tag_size的大小
        self.crf=CRF(label_size)

    def forward(self,input_ids,attention_mask,token_type_ids=None,position_ids=None,head_mask=None,inputs_embeds=None,labels=None):
        """
        这一层就是实现Bert模型的前向传播，然后把输出的logits送给CRF层进行解码，获取损失函数
        """
        #获取bert模型的输出
        bert_out=self.bert_model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,
                                 position_ids=position_ids,head_mask=head_mask,inputs_embeds=inputs_embeds)
        if isinstance(self.bert_model,ElectraModel):
            last_hidden_state, =bert_out
        else:
            last_hidden_state,pooled_output=bert_out

        seq_outs=self.dropout(last_hidden_state)
        logits=self.linear(seq_outs) #将embedd_dim的维度映射到tag的大小的维度
        lengths=attention_mask.sum(axis=1) #获取当前batch内的序列的真实长度
        best_paths=self.crf.get_batch_best_path(logits,lengths)

        if labels is not None:
            #计算损失我们需要忽略[CLS],[SEP],[PAD]的部分
            #logits:[Batch_size,seqlength,embedd_dim] 实际上在序列是[cls_id,xx,xxx,xxx,pad_id,pad_id,...,pad_id]没有sep
            #lengths:
            loss=self.crf.negative_log_loss(inputs=logits[:, 1:, :], length=lengths-1, tags=labels[:, 1:])
            return best_paths,loss
        return best_paths

    def get_bert_tokenizer(self):
        return self.bert_tokenizer
