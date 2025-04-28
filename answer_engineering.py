import csv
import Levenshtein
import re
import pandas as pd
from sklearn import metrics
from sklearn.metrics import precision_score, f1_score, precision_score
import numpy as np

tag_pro_dir = '../data_codereview/data/tag_frequence/tag1_50_pro.csv'
real_answer_dir = '../data_codereview/data/GraphCodeBert_data/question_dev1_set_01.csv'
def get_similar_tag(pre_tag, tag_list):
    res_dict = {}
    for tag in tag_list:
        res_dict[tag] = Levenshtein.distance(tag, pre_tag)

    return min(res_dict, key=res_dict.get)


def answer_engineering(tag_prediction):
    # 答案工程
    tag_prediction_yuan = tag_prediction
    f1 = open(tag_pro_dir, 'r', encoding='utf-8')

    reader1 = csv.reader(f1)
    tag_list = []

    for row in reader1:
        s = row[0]

        tag_list.append(s)
    f1.close()

    for i in range(len(tag_prediction)):
        t = tag_prediction[i]
        for j in range(len(t)):
            s = t[j]
            if '[PAD]' in s:
                s = s.replace(" [PAD]", "")
            tag_prediction[i][j] = s

    for i in range(len(tag_prediction)):

        for j in range(len(tag_prediction[i])):

            tag_ij = tag_prediction[i][j]
           
            state = 1
            list_ij = tag_ij.split(" ")

            if len(list_ij) == 3:

                if tag_ij not in tag_list:
                    list_ij = tag_ij.split(" ")
                    t2 = str(list_ij[0]) + " " + str(list_ij[1])

                    if t2 in tag_list:
                        tag_prediction[i][j] = t2
                    else:
                        list_ij = tag_ij.split(" ")
                        t1 = str(list_ij[0])
                        if t1 in tag_list:
                            tag_prediction[i][j] = t1
                        else:
                            state = 0
            elif len(list_ij) == 2:
                if tag_ij not in tag_list:
                    list_ij = tag_ij.split(" ")
                    t1 = str(list_ij[0])
                    if t1 in tag_list:
                        tag_prediction[i][j] = t1
                    else:
                        state = 0
            else:
                if tag_ij not in tag_list:
                    state = 0

            if state == 0:
                sim_tag = get_similar_tag(tag_ij, tag_list)
                tag_prediction[i][j] = sim_tag
    return tag_prediction


def metrics(tag_pre_new):

    tag_pre_new3 = []
    for i in range(len(tag_pre_new)):
        tag_3 = []
        for j in range(0, 3):
            tag_3.append(tag_pre_new[i][j])
        tag_pre_new3.append(tag_3)

    tag_pre_new5 = []
    for i in range(len(tag_pre_new)):
        tag_5 = []
        for j in range(0, 5):
            tag_5.append(tag_pre_new[i][j])
        tag_pre_new5.append(tag_5)

    f = open(real_answer_dir, 'r', encoding='utf-8')

    reader = csv.reader(f)

    tag_real = []
    for row in reader:
        tag_i = []
        # 取出每条数据的tag，拼成一个list:[a,b,c]
        tag = row[1]
        p = re.compile(r'<(.*?)>')
        for tag_one in p.findall(tag):
            s = tag_one
            if '-' in s:
                s = s.replace("-", " ")
            tag_i.append(s)

        # 所有数据的tag拼成一个list:[[a,b,c],[c,d]]
        tag_real.append(tag_i)
    # print(tag_real)
    pd_tagreal = pd.DataFrame(tag_real)
    pd_tagreal.to_csv('data/tag_real_dev.csv')
    pd_tagpre = pd.DataFrame(tag_pre_new5)
    pd_tagpre.to_csv('data/tag_pre_dev.csv', header = None, index = None)

    pre_correct = []
    pre_correct5 = []
    pre_correct3 = []
    for i in range(len(tag_real)):
        # 遍历数据，i表示第i个数据
        corr_i = 0
        corr_i5 = 0
        corr_i3 = 0
        for j in range(len(tag_real[i])):
            # 遍历第i个数据的第j个tag
            if tag_real[i][j] in tag_pre_new[i]:
                # 判断第i个数据的第j个tag在不在第i个数据预测的tag列表里
                corr_i = corr_i + 1
            if tag_real[i][j] in tag_pre_new5[i]:
                # 判断第i个数据的第j个tag在不在第i个数据预测的tag列表里
                corr_i5 = corr_i5 + 1
            if tag_real[i][j] in tag_pre_new3[i]:
                # 判断第i个数据的第j个tag在不在第i个数据预测的tag列表里
                corr_i3 = corr_i3 + 1
        pre_correct.append(corr_i)
        pre_correct5.append(corr_i5)
        pre_correct3.append(corr_i3)

    # @3
    sum = 0
    for k in range(len(pre_correct3)):
        sum = sum + int(pre_correct3[k])
    prediction_3 = sum / (3 * len(pre_correct3))
    # @5
    sum = 0
    for k in range(len(pre_correct5)):
        sum = sum + int(pre_correct5[k])
    prediction_5 = sum / (5 * len(pre_correct5))

    # @10
    sum = 0
    for k in range(len(pre_correct)):
        sum = sum + int(pre_correct[k])
    prediction_10 = sum / (10 * len(pre_correct))

    # @3
    sum = 0
    rec_sub = 0
    for k in range(len(pre_correct3)):
        rec_sub = rec_sub + pre_correct3[k] / len(tag_real[k])

    recall_3 = rec_sub / len(pre_correct3)
    # @5
    sum = 0
    rec_sub = 0
    for k in range(len(pre_correct5)):
        rec_sub = rec_sub + pre_correct5[k] / len(tag_real[k])

    recall_5 = rec_sub / len(pre_correct5)

    # @10
    sum = 0
    rec_sub = 0
    for k in range(len(pre_correct)):
        rec_sub = rec_sub + pre_correct[k] / len(tag_real[k])

    recall_10 = rec_sub / len(pre_correct)

    # @3
    sum = 0
    for k in range(len(pre_correct3)):
        rec_i = pre_correct3[k] / len(tag_real[k])
        pre_i = int(pre_correct3[k]) / 3

        if (pre_i + rec_i) == 0:
            f1_i = 0
        else:
            f1_i = 2 * pre_i * rec_i / (pre_i + rec_i)

        sum += f1_i

    f1_3 = sum / len(pre_correct3)

    # @5
    sum = 0
    for k in range(len(pre_correct5)):
        rec_i = pre_correct5[k] / len(tag_real[k])
        pre_i = int(pre_correct5[k]) / 5

        if (pre_i + rec_i) == 0:
            f1_i = 0
        else:
            f1_i = 2 * pre_i * rec_i / (pre_i + rec_i)

        sum += f1_i

    f1_5 = sum / len(pre_correct5)

    # @10
    sum = 0
    for k in range(len(pre_correct)):
        rec_i = pre_correct[k] / len(tag_real[k])
        pre_i = int(pre_correct[k]) / 10

        if (pre_i + rec_i) == 0:
            f1_i = 0
        else:
            f1_i = 2 * pre_i * rec_i / (pre_i + rec_i)

        sum += f1_i

    f1_10 = sum / len(pre_correct)

    print('******************** 指标 ********************')
    print('******************** TOP 3 ********************')
    print('Precise@3 ', prediction_3)
    print('Recall@3: ', recall_3)
    print('F1@3: ', f1_3)
    print('******************** TOP 5 ********************')
    print('Precise@5: ', prediction_5)
    print('Recall@5: ', recall_5)
    print('F1@5: ', f1_5)
    print('******************** TOP 10 ********************')
    print('Precise@10: ', prediction_10)
    print('Recall@10: ', recall_10)
    print('F1@10: ', f1_10)
    
    
def question_engineering(tag_prediction):
    # 答案工程
    tag_list = ['good', 'general', 'bad']
#     print(tag_prediction)
    for i in range(len(tag_prediction)):
        if tag_prediction[i] not in tag_list:
            tag_ij = tag_prediction[i]
            sim_tag = get_similar_tag(tag_ij, tag_list)
            tag_prediction[i] = sim_tag
    
    return tag_prediction
def question_metrics(tag_pre):
    f = open('../data_codereview/data/GraphCodeBert_data/question_dev1_set_01.csv', 'r', encoding='utf-8')

    reader = csv.reader(f)

    tag_real = []
    for row in reader:
        tag_real.append(row[5])
    
    pre_ans = []
    state = 0
    question_label = 0
    question_pre = []
    print(len(tag_pre))
    print(len(tag_real))
    for i in range(len(tag_pre)):
        if tag_pre[i] == tag_real[i]:
            if tag_pre[i] == 'good':
                question_label = 1
            elif tag_pre[i] == 'bad':
                question_label = 0
            state = 1
        else:
            state = 0
        pre_ans.append(state)
        question_pre.append(question_label)
        
    question_pre = np.array(question_pre)
    question_real = []
    question_label = 0
    for i in range(len(tag_real)): 
        if tag_pre[i] == 'good':
            question_label = 1
        elif tag_pre[i] == 'bad':
            question_label = 0
        question_real.append(question_label)
    question_real = np.array(question_real)
    
    acc1 = precision_score(question_real, question_pre, average="micro")
    
    f1_0 = f1_score(question_pre==0,question_real==0,labels=True)
    f1_1 = f1_score(question_pre==1,question_real==1,labels=True)
    

    
        
        
        
    all_right = 0
    for i in range(len(pre_ans)):
        all_right += pre_ans[i]
    acc = all_right / len(pre_ans)
    print("################    Question   ################")
    print("Acc: ", acc)
    print("Acc1: ", acc1)
    print("F1 bad:", f1_0)
    print("F1 good:", f1_1)
    
    
    
        
    