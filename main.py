import argparse
import multiprocessing
import pandas as pd
import torch
from transformers import BertTokenizer, BertForMaskedLM, BertModel
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaForMaskedLM, RobertaTokenizer,BertModel)
from datasets import load_dataset
# 超参数
from myDataset import convert_examples_to_features
from model import BertForMaskedLM1, DataCollatorForPrompt, Model
from myTokenizers import tokenize_function
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
# import os
# os.environ['CUDA_VISIBLE_DEVICES']='1'


def main():
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
    device = torch.device(CONFIG['cuda'] if torch.cuda.is_available() else "cpu")
#     device = torch.device("cpu")

    if CONFIG['do_train']:
        graphCodeBertModel = RobertaModel.from_pretrained(CONFIG['model_name_or_path'])

        graphCodeBertModel.to(device)
        # 加载PLM和tokenizer
        model_name = "Pretrained_LMs/bert-base-cased2"
        BertModel = BertForMaskedLM1.from_pretrained(model_name)
        # 补充实验，PLM换成彼此的那个
#         bertmodel = torch.load("Train_Model/question_plm/epoch_5", map_location = torch.device('cpu'))
#         BertModel.load_state_dict(torch.load("Train_Model/question_plm/epoch_5", map_location = torch.device('cpu')), False)
#         BertModel.load_state_dict(torch.load("Train_Model/cr_plm/epoch_5", map_location = torch.device('cpu')), False)

#         BertModel = BertForMaskedLM1.from_pretrained("Train_Model/newdata_codelen150/epoch_5")
        # 分词要用加入词表之后的模型
        bertTokenizer = BertTokenizer.from_pretrained("./Pretrained_LMs/bert-base-cased2", padding=True, truncation=True)
        BertModel.to(device)

        model = Model(BertModel, graphCodeBertModel)
    if CONFIG['do_eval']:
        bertTokenizer = BertTokenizer.from_pretrained("Pretrained_LMs/bert-base-cased2", padding=True, truncation=True)
        # model =
#         model = torch.load('Train_Model/question_01/epoch_4', map_location={'cuda:3': 'cuda:3'})
#         model = torch.load('Train_Model/ablation/epoch_5', map_location={'cuda:3': 'cuda:3'})
#         model = torch.load('Train_Model/newdata_codelen150/epoch_5', map_location={'cuda:2': 'cuda:2'})
        # model=torch.load('Train_Model/EXP6/graph_with_grad-language-output0-cs1/codelen120-labeladd/epoch_7', map_location={'cuda:3': 'cuda:2'})
        # 加实验
#         model = torch.load('Train_Model/add-useques-asplm1/epoch_6', map_location={'cuda:0': 'cuda:3'})
#         prompt设计实验
        model = torch.load(CONFIG['dev_model_path'], map_location={'cuda:3': 'cuda:1'})
        model.to(device)

    # 加载数据
    if CONFIG['do_train']:
#         2023
#         data_dir = "../data_codereview/data/GraphCodeBert_data/"
#         data_files = {"train": data_dir + "train1_set.csv"}
#         question 3分类
#         data_files = {"train": data_dir + "question_train1_set1.csv"}
#         data_files = {"train": data_dir + "question_train1_set_01.csv"}
#         设计实验:
        data_files = {"train": CONFIG['train_data_file']}
    if CONFIG['do_eval']:
#         data_dir = "../data_codereview/data/GraphCodeBert_data/"
#         data_files = {"test": data_dir + "question_dev1_set_01.csv"}
#         data_files = {"test": data_dir + "dev1_set.csv"}
        # 设计实验：
        data_files = {"test": CONFIG['dev_data_file']}

    raw_datasets = load_dataset("csv", data_files=data_files, sep=",", header=None)
    # 编码
    # dataset 的批量处理,map相当于对数据集进行运算，运算后的结果会在数据集后面进行添加
    # 如果只需要处理后的数据，那么在remove_columns就可以添加需要删除的列的名字。
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=8,  # 多个进程一起编码
        remove_columns=['0', '1', '2', '3', '4', '5'],
        desc="Running tokenizer on dataset line_by_line",
    )
    #
    pool = multiprocessing.Pool(16)
    # codedata=pd.read_csv(CONFIG['train_data_file']).iloc[:,2]
    # examples=pool.map(convert_examples_to_features,tqdm(codedata, total=len(codedata)))

#     start_index = 5
#     end_index = 8
    start_index = CONFIG['start_index']
    end_index = CONFIG['end_index']
    start_question_index = CONFIG['start_question_index']
    end_question_index = CONFIG['end_question_index']
   
    dc = DataCollatorForPrompt(
        tokenizer=bertTokenizer,
        start_index=start_index,
        end_index=end_index,
        start_question_index=start_question_index,
        end_question_index=end_question_index
    )

   
    if CONFIG['do_train']:
        train_dataloader = DataLoader(tokenized_datasets['train'], collate_fn=dc, batch_size=CONFIG['train_batch_size'],
                                      num_workers=0)
        # question 分类里边把weight_decay=1e-2
        optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'], eps=1e-8, weight_decay=1e-2)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                    num_training_steps=len(train_dataloader) * CONFIG[
                                                        'num_train_epochs'])
    if CONFIG['do_eval']:
        test_dataloader = DataLoader(tokenized_datasets['test'], collate_fn=dc, batch_size=CONFIG['eval_batch_size'],
                                     num_workers=0)
        optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'], eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                    num_training_steps=len(test_dataloader) * CONFIG[
                                                        'num_train_epochs'])

    epochs = CONFIG['num_train_epochs']

    if CONFIG['do_train']:
        train(model, train_dataloader, optimizer, scheduler, device, pool)

    if CONFIG['do_eval']:
        # 获取预测结果
#         tag_prediction = evaluate(model, test_dataloader, optimizer, device, pool)


#         tag_prediction_ae = answer_engineering(tag_prediction)

#         # print(tag_prediction_ae)
#         pd_tagpre = pd.DataFrame(tag_prediction_ae)
#         # pd_tagpre.to_csv('data/tag_prediction_EXP6_t220_epoch5.csv')
#         # 计算指标并输出
#         metrics(tag_prediction_ae)
        
        '''question分类'''
        tag_prediction, question = evaluate(model, test_dataloader, optimizer, device, pool)
        tag_prediction_q = question_engineering(question)
        question_metrics(tag_prediction_q)
        
#         print(tag_prediction)
        tag_prediction_ae = answer_engineering(tag_prediction)
        pd_tagpre = pd.DataFrame(tag_prediction_ae)
        metrics(tag_prediction_ae)


if __name__ == '__main__':
    main()
