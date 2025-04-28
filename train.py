import torch
from transformers import BertTokenizer, BertForMaskedLM, AdamW, BertModel
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
from datasets import load_dataset
# 超参数
from model import BertForMaskedLM1, DataCollatorForPrompt
from myTokenizers import tokenize_function
from torch.utils.data import DataLoader
from config import CONFIG
from myDataset import getcodeAttenMask


# def train(model, iterator, optimizer, scheduler, device, pool):
#     # 将所有模型参数的梯度设置为零
#     model.zero_grad()
#     # 将模型设置为训练模式
#     model.train()
#     # graphCodeBertModel.train()
#     text_model = BertModel.from_pretrained("Pretrained_LMs/bert-base-cased2")
#     code_model = RobertaModel.from_pretrained("./microsoft/graphcodebert-base1")
#     text_model.to(device)
#     code_model.to(device)
#     epoch_loss = 0
#     for epoch in range(CONFIG['epoch']):
#         # optimizer.zero_grad()
#         for i, batch in enumerate(iterator):
#             # get inputs
#             token_type_ids = batch["token_type_ids"]
#             attention_mask = batch["attention_mask"]
#             input_ids = batch["input_ids"]
#             # codetext = batch["code"]
#             labels = batch["labels"]

#             code_inputs = batch['code_ids']
#             code_attn_mask = batch['code_attn_mask']
#             position_idx = batch['position_idx']
#             code_identity = batch['code_identity']

#             code_token = batch['code_token']

#             # 需要 LongTensor
#             input_ids, token_type_ids, labels, attention_mask = input_ids.long(), token_type_ids.long(), labels.long(), attention_mask.long()
#             # code_inputs,code_attn_mask,position_idx = code_inputs.long(),code_attn_mask.long(),position_idx.long()
#             code_identity = code_identity.long()

#             # 梯度清零
#             optimizer.zero_grad()

#             input_ids, token_type_ids, labels, attention_mask = input_ids.to(device), token_type_ids.to(
#                 device), labels.to(device), attention_mask.to(device)
#             code_inputs, code_attn_mask, position_idx = code_inputs.to(device), code_attn_mask.to(
#                 device), position_idx.to(device)
#             code_identity = code_identity.to(device)

#             code_token = code_token.long().to(device)


#             with torch.no_grad():  
#                 code_token_embed = code_model(code_token)[0]
#                 text_token_embed = text_model(input_ids)[0]

#             # 前向传播, 可更新部分
#             code_embeddings = model(code_inputs, code_attn_mask, position_idx)
#             # print(len(code_embeddings))

#             # code_list = feature.cpu().detach().numpy().tolist() 
#             code_list = code_embeddings  

#             output = model.bertEncoder(
#                 code_identity=code_identity,
#                 code=code_list,
#                 text=text_token_embed,
#                 code_token=code_token_embed,
#                 input_ids=input_ids,
#                 token_type_ids=token_type_ids,
#                 labels=labels,
#                 attention_mask=attention_mask
#             )

#             loss = output[0]
#             y_pred_prob = output[1]

#             # 反向传播
#             loss.backward()
#             # if (i + 1) % 2 == 0:
#             #     optimizer.step()
#             #     scheduler.step()
#             #     optimizer.zero_grad()
#             optimizer.step()
#             scheduler.step()
#             # epoch 中的 loss 累加
#             epoch_loss += loss.item()
#             if i % 200 == 0:
#                 print("current loss:", epoch_loss / (i + 1))

#         print('epoch:', epoch)
#         # question 三分类
# #         torch.save(model, 'Train_Model/question_multiprompt/epoch_' + str(epoch + 1))
# #         torch.save(model, 'Train_Model/ablation/epoch_' + str(epoch + 1))
# #         torch.save(model, 'Train_Model/add-useques-asplm1/epoch_' + str(epoch + 1))
# #         torch.save(model, 'Train_Model/add-usecr-asplm/epoch_' + str(epoch + 1))

#         #加实验
# #         torch.save(model.state_dict(), 'Train_Model/question_plm/epoch_' + str(epoch + 1))
# #         torch.save(model.state_dict(), 'Train_Model/cr_plm/epoch_' + str(epoch + 1))

#         #设计实验
#         torch.save(model, CONFIG['train_model_path'] + str(epoch + 1))
#     return epoch_loss / len(iterator)


# def train(model, iterator, optimizer, scheduler, device, pool):
#     # 将所有模型参数的梯度设置为零
#     model.zero_grad()
#     # 将模型设置为训练模式
#     model.train()
#     # graphCodeBertModel.train()
#     text_model = BertModel.from_pretrained("Pretrained_LMs/bert-base-cased2")
#     code_model = RobertaModel.from_pretrained("./microsoft/graphcodebert-base1")
#     text_model.to(device)
#     code_model.to(device)
#     epoch_loss = 0
#     for epoch in range(CONFIG['epoch']):
#         # optimizer.zero_grad()
#         for i, batch in enumerate(iterator):
#             # get inputs
#             token_type_ids = batch["token_type_ids"]
#             attention_mask = batch["attention_mask"]
#             input_ids = batch["input_ids"]
#             # codetext = batch["code"]
#             labels = batch["labels"]

#             code_inputs = batch['code_ids']
#             code_attn_mask = batch['code_attn_mask']
#             position_idx = batch['position_idx']
#             code_identity = batch['code_identity']

#             code_token = batch['code_token']

#             # 需要 LongTensor
#             input_ids, token_type_ids, labels, attention_mask = input_ids.long(), token_type_ids.long(), labels.long(), attention_mask.long()
#             # code_inputs,code_attn_mask,position_idx = code_inputs.long(),code_attn_mask.long(),position_idx.long()
#             code_identity = code_identity.long()

#             # 梯度清零
#             optimizer.zero_grad()
#             # 迁移到GPU
#             input_ids, token_type_ids, labels, attention_mask = input_ids.to(device), token_type_ids.to(
#                 device), labels.to(device), attention_mask.to(device)
#             code_inputs, code_attn_mask, position_idx = code_inputs.to(device), code_attn_mask.to(
#                 device), position_idx.to(device)
#             code_identity = code_identity.to(device)

#             code_token = code_token.long().to(device)


#             with torch.no_grad():  
#                 code_token_embed = model(code_inputs, code_attn_mask, position_idx)
#                 text_token_embed = text_model(input_ids)[0]

#             code_embeddings = code_model(code_token)[0]
#             # print(len(code_embeddings))
#             # code_list = feature.cpu().detach().numpy().tolist() 
#             code_list = code_embeddings 

#             output = model.bertEncoder(
#                 code_identity=code_identity,
#                 code=code_list,
#                 text=text_token_embed,
#                 code_token=code_token_embed,
#                 input_ids=input_ids,
#                 token_type_ids=token_type_ids,
#                 labels=labels,
#                 attention_mask=attention_mask
#             )

#             loss = output[0]
#             y_pred_prob = output[1]

#             # 反向传播
#             loss.backward()
#             # if (i + 1) % 2 == 0:
#             #     optimizer.step()
#             #     scheduler.step()
#             #     optimizer.zero_grad()
#             optimizer.step()
#             scheduler.step()
#             # epoch 中的 loss 累加
#             epoch_loss += loss.item()
#             if i % 200 == 0:
#                 print("current loss:", epoch_loss / (i + 1))

#         print('epoch:', epoch)
#         # question 三分类
# #         torch.save(model, 'Train_Model/question_multiprompt/epoch_' + str(epoch + 1))
# #         torch.save(model, 'Train_Model/ablation/epoch_' + str(epoch + 1))
# #         torch.save(model, 'Train_Model/add-useques-asplm1/epoch_' + str(epoch + 1))
# #         torch.save(model, 'Train_Model/add-usecr-asplm/epoch_' + str(epoch + 1))

#         #加实验
# #         torch.save(model.state_dict(), 'Train_Model/question_plm/epoch_' + str(epoch + 1))
# #         torch.save(model.state_dict(), 'Train_Model/cr_plm/epoch_' + str(epoch + 1))

#         # 设计实验
#         torch.save(model, CONFIG['train_model_path'] + str(epoch + 1))
#     return epoch_loss / len(iterator)

def train(model, iterator, optimizer, scheduler, device, pool):
    # 将所有模型参数的梯度设置为零
    model.zero_grad()
    # 将模型设置为训练模式
    model.train()
    # graphCodeBertModel.train()
    text_model = BertModel.from_pretrained("Pretrained_LMs/bert-base-cased2")
    code_model = RobertaModel.from_pretrained("./microsoft/graphcodebert-base1")
    text_model.to(device)
    code_model.to(device)
    epoch_loss = 0
    for epoch in range(CONFIG['epoch']):
        # optimizer.zero_grad()
        for i, batch in enumerate(iterator):
            # get inputs
            token_type_ids = batch["token_type_ids"]
            attention_mask = batch["attention_mask"]
            input_ids = batch["input_ids"]
            # codetext = batch["code"]
            labels = batch["labels"]

            code_inputs = batch['code_ids']
            code_attn_mask = batch['code_attn_mask']
            position_idx = batch['position_idx']
            code_identity = batch['code_identity']

            code_token = batch['code_token']
            wiki_input_ids = batch['wiki_input_ids']

            # 需要 LongTensor
            input_ids, token_type_ids, labels, attention_mask = input_ids.long(), token_type_ids.long(), labels.long(), attention_mask.long()
            # code_inputs,code_attn_mask,position_idx = code_inputs.long(),code_attn_mask.long(),position_idx.long()
            code_identity = code_identity.long()
            wiki_input_ids = wiki_input_ids.long()

            # 梯度清零
            optimizer.zero_grad()
            # 迁移到GPU
            input_ids, token_type_ids, labels, attention_mask = input_ids.to(device), token_type_ids.to(
                device), labels.to(device), attention_mask.to(device)
            code_inputs, code_attn_mask, position_idx = code_inputs.to(device), code_attn_mask.to(
                device), position_idx.to(device)
            code_identity = code_identity.to(device)
            wiki_input_ids = wiki_input_ids.to(device)

            code_token = code_token.long().to(device)


            with torch.no_grad():  
                code_token_embed = model(code_inputs, code_attn_mask, position_idx) 
                text_token_embed = text_model(input_ids)[0]


            code_embeddings = text_model(wiki_input_ids)[0]
            # print(len(code_embeddings))
            # code_list = feature.cpu().detach().numpy().tolist() 
            code_list = code_embeddings  

            # 获取模型输出
                output = model.bertEncoder(
                code_identity=code_identity,
                code=code_list,
                text=text_token_embed,
                code_token=code_token_embed,
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                labels=labels,
                attention_mask=attention_mask
            )

            loss = output[0]
            y_pred_prob = output[1]

            # 反向传播
            loss.backward()
            # if (i + 1) % 2 == 0:
            #     optimizer.step()
            #     scheduler.step()
            #     optimizer.zero_grad()
            optimizer.step()
            scheduler.step()
            # epoch 中的 loss 累加
            epoch_loss += loss.item()
            if i % 200 == 0:
                print("current loss:", epoch_loss / (i + 1))

        print('epoch:', epoch)
        # question 三分类
#         torch.save(model, 'Train_Model/question_multiprompt/epoch_' + str(epoch + 1))
#         torch.save(model, 'Train_Model/ablation/epoch_' + str(epoch + 1))
#         torch.save(model, 'Train_Model/add-useques-asplm1/epoch_' + str(epoch + 1))
#         torch.save(model, 'Train_Model/add-usecr-asplm/epoch_' + str(epoch + 1))

        #加实验
#         torch.save(model.state_dict(), 'Train_Model/question_plm/epoch_' + str(epoch + 1))
#         torch.save(model.state_dict(), 'Train_Model/cr_plm/epoch_' + str(epoch + 1))

        # 设计实验
        torch.save(model, CONFIG['train_model_path'] + str(epoch + 1))
    return epoch_loss / len(iterator)

