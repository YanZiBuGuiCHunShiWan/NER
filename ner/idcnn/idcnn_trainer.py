import torch
import json,math
from torch.optim import Adam
from torch import LongTensor
from ner.idcnn.idcnn_crf import IDCNNCRF
from ner.base.base_trainer import  BaseTrainer
from ner.vocab import Vocab
from loggers import logger
from sklearn.utils import shuffle

class IDCNNCRFTrainer(BaseTrainer):
        def __init__(self,model_dir,filters,hidden_num,embedd_dim,dropout_rate,learning_rate=1e-3,ckpt_name="idcnn_crf.bin",
                     vocab_name="Vocab.json",load_last_ckpt=True):
            self.model_dir=model_dir
            self.ckpt_name = ckpt_name
            self.vocab_name = vocab_name
            self.load_last_ckpt = load_last_ckpt
            self.filters=filters
            self.hidden_num=hidden_num
            self.embedd_dim=embedd_dim
            self.dropout_rate=dropout_rate
            self.learning_rate = learning_rate
            self.batch_size = None
            self.epoch = None
            self.max_len = None
            self.device="cuda" if torch.cuda.is_available() else "cpu"
            self.vocab=Vocab()

        def _build_model(self):
            """
            构建IDCNN_CRF 的模型
            :return:
            """
            self.model=IDCNNCRF(seqlen=self.max_len,filters=self.filters,hidden_num=self.hidden_num,vocab_size=self.vocab.vocab_size,
                                embedd_dim=self.embedd_dim,label_size=self.vocab.label_size,dropout_rate=self.dropout_rate)
            if self.load_last_ckpt:
                self.model.load_state_dict(torch.load("{}/{}".format(self.model_dir,self.ckpt_name),map_location=self.device))
            self.optimizer=Adam(self.model.parameters(),lr=self.learning_rate,weight_decay=1e-4)
            self.model.to(self.device)

        def _save_config(self):
            """
            保存当前模型训练参数设置的配置
            """
            config={
                'max_len': self.max_len,
                'filters': self.filters,
                'hidden_num': self.hidden_num,
                'embedding_size': self.embedd_dim,
                'dropout_rate': self.dropout_rate,
                'vocab_size': self.vocab.vocab_size,
                'label_size': self.vocab.label_size,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'epoch': self.epoch,
                'ckpt_name': self.ckpt_name,
                'vocab_name': self.vocab_name
            }
            with open('{}/train_config.json'.format(self.model_dir), 'w') as f:
                f.write(json.dumps(config, indent=4))

        def _transform_batch(self,batch_texts,batch_labels,max_length):
            """将batch的文本及labels转换为id的tensor形式"""
            batch_input_ids, batch_label_ids, input_lens = [], [], []
            for text, labels in zip(batch_texts, batch_labels):  # 迭代每条text, labels
                batch_input_ids.append([]), batch_label_ids.append([])
                assert len(text) == len(labels)  # 确保等长
                input_lens.append(len(labels))  # 更新input_lens
                for seg, label in zip(text, labels):  # 迭代每个位置的char和label
                    # 通过vocab，将char和label都转化为id
                    batch_input_ids[-1].append(self.vocab.vocab2id.get(seg, self.vocab.unk_vocab_id))
                    batch_label_ids[-1].append(self.vocab.tag2id[label])
                # pad到max_length
                batch_input_ids[-1] += [self.vocab.pad_vocab_id] * (max_length - len(batch_input_ids[-1]))
                batch_label_ids[-1] += [self.vocab.pad_tag_id] * (max_length - len(batch_label_ids[-1]))
                # 转为tensor
            batch_input_ids, batch_label_ids, input_lens =LongTensor(batch_input_ids), LongTensor(batch_label_ids), LongTensor(input_lens)

            # 将数据拷贝到当前设备
            batch_input_ids, batch_label_ids, input_lens =batch_input_ids.to(self.device), batch_label_ids.to(self.device), input_lens.to(self.device)

            return batch_input_ids, batch_label_ids, input_lens


        def train(self,train_texts,labels,validate_texts,validate_labels,batch_size=30,epoches=10,max_len=200):
            """训练
                Args:
                    train_texts: list[list[str]].训练样本
                    labels: list[list[str]].标签
                    validate_texts: list[list[str]].验证样本
                    validate_labels: list[list[str]].验证集标签
                    batch_size: int
                    epoches: int
                    max_len:
            """
            # 将train函数的一些参数更新到对象
            self.batch_size = batch_size
            self.epoches = epoches
            self.max_len = max_len
            self.vocab.build_vocab(texts=train_texts, labels=labels)  # 构建词库
            self._build_model()  # 构建模型
            # 保存词库和config
            self.vocab.save_vocab('{}/{}'.format(self.model_dir, self.vocab_name))
            self._save_config()

            logger.info('train samples: {}, validate samples: {}'.format(len(train_texts), len(validate_texts)))
            best_loss = float("inf")
            loss_buff = []  # 缓存最近的10个valid loss
            max_loss_num = 10  # 按照最新10个valid平均loss保存模型
            step = 0
            for epoch in range(self.epoches):
                for bacth_idx in range(math.ceil(len(train_texts)/batch_size)):
                    text_batch=train_texts[batch_size*bacth_idx:batch_size*(bacth_idx+1)]
                    labels_batch=labels[batch_size*bacth_idx:batch_size*(bacth_idx+1)]
                    step+=1
                    self.model.train()
                    self.model.zero_grad()
                    batch_input_ids, bacth_label_ids,input_lens=self._transform_batch(text_batch,labels_batch,self.max_len)
                    best_path,loss=self.model(batch_input_ids,input_lens,labels=bacth_label_ids)
                    loss.backward()
                    self.optimizer.step()

                    train_acc=self._get_acc_one_step(best_path,bacth_label_ids,input_lens)
                    validate_acc,validate_loss=self.validate(validate_texts,validate_labels)
                    loss_buff.append(validate_loss)
                    if len(loss_buff)>max_loss_num:
                        loss_buff=loss_buff[-max_loss_num:]
                    avg_loss=sum(loss_buff)/max_loss_num if len(loss_buff)==max_loss_num else None

                    logger.info("epoch % d,step %d,train_loss %.4f,train_acc %.4f,validate_loss %.4f,validate_acc %.4f,"
                                "last %d ,avg valid loss %s"%(epoch,step,loss,train_acc,validate_loss,validate_acc,max_loss_num,'%.4f' % avg_loss if avg_loss else avg_loss))

                    if avg_loss and avg_loss < best_loss:
                        best_loss = avg_loss
                        torch.save(self.model.state_dict(), '{}/{}'.format(self.model_dir, self.ckpt_name))
                        logger.info("model saved")


        def validate(self,validate_texts,validate_labels,sample_size=100):
            self.model.eval()
            batch_texts,batch_labels=[return_val[:sample_size] for return_val in shuffle(validate_texts,validate_labels)]
            with torch.no_grad():
                bacth_texts_id,bacth_labels_id,input_lens=self._transform_batch(batch_texts,batch_labels,self.max_len)

                best_path,loss=self.model(bacth_texts_id,input_lens,labels=bacth_labels_id)

                acc=self._get_acc_one_step(best_path,bacth_labels_id,input_lens)
                return acc,loss

        def _get_acc_one_step(self,best_path,labels_batch,input_lengths):
            total,correct=0,0.0
            for predict_labels,labels,input_lengths in zip(best_path,labels_batch,input_lengths.tolist()):
                total+=input_lengths
                correct+=(predict_labels[:input_lengths]==labels[:input_lengths].cpu()).int().sum().item()
            accuracy=correct/total
            return float(accuracy)