from utils import evaluate
from ner.idcnn.idcnn_trainer import IDCNNCRFTrainer
from ner.idcnn.idcnn_predictor import IDCNNPredictor

def readconll(file_path):
    with open(file_path,encoding="utf-8",errors="ignore") as f:
        texts,labels=[],[]
        for example in f.read().split("\n\n"):
            example=example.strip()
            if not  example:
                continue
            texts.append([]),labels.append([])
            for term in example.split("\n"):
                if len(term.split("\t")) !=2: #跳过不合法的行
                    continue
                char,label=term.split("\t")
                texts[-1].append(char),labels[-1].append((label))
        return texts,labels
#读取数据
train_texts,train_labels=readconll("../../data/Weibo/weiboNER.conll.train")
vali_texts,vali_labels=readconll("../../data/Weibo/weiboNER.conll.dev")
test_texts,test_labels=readconll("../../data/Weibo/weiboNER.conll.test")

#实例化trainer,并设置参数
trainer=IDCNNCRFTrainer(model_dir="../../models/idcnn", filters=64, hidden_num=256, embedd_dim=128, dropout_rate=0.3, learning_rate=1e-3,
                        load_last_ckpt=False)
trainer.train(train_texts=train_texts,labels=train_labels,validate_texts=vali_texts,validate_labels=vali_labels,
              batch_size=32,epoches=5,max_len=256)

# 实例化predictor，加载模型
predictor = IDCNNPredictor('../../models/idcnn')
predict_labels = predictor.predict(test_texts, batch_size=20)

# 将结果输出为3列
out = open('./tmp/dev_results.txt', 'w', encoding='utf-8')
for text, each_true_labels, each_predict_labels in zip(test_texts, test_labels, predict_labels):
    for char, true_label, predict_label in zip(text, each_true_labels, each_predict_labels):
        out.write('{}\t{}\t{}\n'.format(char, true_label, predict_label))
    out.write('\n')
out.close()

# 评估
evaluate.eval('tmp/dev_results.txt')