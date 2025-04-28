import argparse
import multiprocessing
import pandas as pd
import torch
from transformers import BertTokenizer, BertForMaskedLM, BertModel
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaForMaskedLM, RobertaTokenizer)
from datasets import load_dataset
# 超参数
from myDataset import convert_examples_to_features
from model import BertForMaskedLM1, DataCollatorForPrompt, Model
from selfTokenizers import tokenize_function
from torch.utils.data import DataLoader
import torch
from train import train
from utils import set_seed
from config import CONFIG
from tqdm import tqdm, trange
from test import *
from answer_engineering import *

"""
train：  训练数据格式：
         |0      |1       |2       |3               |4               |5
         |text   |label   |code    |code_language   |code_identity   |id
"""


set_seed(CONFIG['seed'])
# learning_rate = 1e-5
# weight_decay = 1e-2
# epochs = 6
# batch_size = 8
# max_len = 300
#
# # model_name = "Pretrained_LMs/bert-base-cased"
# # model_name = "hfl/chinese-roberta-wwm-ext"
#
# maxlen_c = 80
# maxlen_t = 250

# 模型初始化
# 定义device

# if CONFIG['wandb']:
#     import wandb
#     wandb.init(project="tagRec", entity="graphCodeBert")
device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device1 = torch.device("cpu")

if CONFIG['do_train']:
    graphCodeBertModel = RobertaModel.from_pretrained(CONFIG['model_name_or_path'])

    graphCodeBertModel.to(device1)
    # 加载PLM和tokenizer
    model_name = "bert-base-uncased"
    BertModel = BertForMaskedLM1.from_pretrained(model_name)
    # 分词要用加入词表之后的模型
    bertTokenizer = BertTokenizer.from_pretrained("Pretrained_LMs/bert-base-cased1", padding=True, truncation=True)
    BertModel.to(device1)

    model = Model(BertModel, graphCodeBertModel)
if CONFIG['do_eval']:
    bertTokenizer = BertTokenizer.from_pretrained("Pretrained_LMs/bert-base-cased1", padding=True, truncation=True)
    # model = torch.load('Train_Model/EXP6/graph_with_grad-language-output0-cs/epoch_1')

    model.load_state_dict(torch.load('Train_Model/EXP6/graph_with_grad-language-output0-cs/epoch_1', map_location={'cuda:2': 'cuda:0'}))
    model.to(device1)

# 加载数据
if CONFIG['do_train']:
    data_dir = "../data_codereview/data/GraphCodeBert_data/"
    data_files = {"train": data_dir + "train_set_1.csv"}
if CONFIG['do_eval']:
    data_dir = "../data_codereview/data/GraphCodeBert_data/"
    data_files = {"test": data_dir + "dev_set_1.csv"}

raw_datasets = load_dataset("csv", data_files=data_files, sep=",", header=None)
# 编码
# dataset 的批量处理,map相当于对数据集进行运算，运算后的结果会在数据集后面进行添加
# 如果只需要处理后的数据，那么在remove_columns就可以添加需要删除的列的名字。
tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    num_proc=4,  # 多个进程一起编码
    remove_columns=['0', '1', '2', '3', '4', '5'],
    desc="Running tokenizer on dataset line_by_line",
)
#
pool = multiprocessing.Pool(16)
# codedata=pd.read_csv(CONFIG['train_data_file']).iloc[:,2]
# examples=pool.map(convert_examples_to_features,tqdm(codedata, total=len(codedata)))

start_index = 5
end_index = 8
# 使用DataCollatorForPrompt，并传入起始位置和结束位置
# collate--整理
# dc-数据整理,collate_fn可理解为函数句柄、指针...或者其他可调用类(实现__call__函数)。 函数输入为list，list中的元素为欲取出的一系列样本。
dc = DataCollatorForPrompt(
    tokenizer=bertTokenizer,
    start_index=start_index,
    end_index=end_index
)

# 使用dataloader加载数据
# params = {"batch_size": batch_size, "shuffle": True, "num_workers": 2}
# collate-fn函数就是手动将抽取出的样本堆叠起来的函数
# 通过collate_fn函数可以对这些样本做进一步的处理(任何你想要的处理)，原则上返回值应当是一个有结构的batch。而DataLoader每次迭代的返回值就是collate_fn的返回值。
if CONFIG['do_train']:
    train_dataloader = DataLoader(tokenized_datasets['train'], collate_fn=dc, batch_size=CONFIG['train_batch_size'],
                                  num_workers=0)
    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'], eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=len(train_dataloader) * CONFIG[
                                                    'num_train_epochs'])
if CONFIG['do_eval']:
    test_dataloader = DataLoader(tokenized_datasets['test'], collate_fn=dc, batch_size=CONFIG['train_batch_size'],
                                 num_workers=0)
    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'], eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=len(test_dataloader) * CONFIG[
                                                    'num_train_epochs'])
# get optimizer and scheduler

# 定义优化器
# no_decay = ['bias', 'LayerNorm.weight']
# optimizer_grouped_parameters = [
#         {'params': [p for n, p in BertModel.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
#         {'params': [p for n, p in BertModel.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
#         {'params': [p for n, p in graphCodeBertModel.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
#         {'params': [p for n, p in graphCodeBertModel.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
#
# optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

# 训练模型
# model(**x)
epochs = CONFIG['num_train_epochs']

if CONFIG['do_train']:
    train(model, train_dataloader, optimizer, scheduler, device1, pool)

if CONFIG['do_eval']:
    # 获取预测结果
    tag_prediction = evaluate(model, test_dataloader, optimizer, device1, pool)

    # 将预测的标签进行答案工程
    tag_prediction_ae = answer_engineering(tag_prediction)

    # print(tag_prediction_ae)
    pd_tagpre = pd.DataFrame(tag_prediction_ae)
    # pd_tagpre.to_csv('data/tag_prediction_EXP6_t220_epoch5.csv')
    # 计算指标并输出
    metrics(tag_prediction_ae)

