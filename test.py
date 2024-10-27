import torch
from transformers import BertTokenizer,RobertaModel, BertModel

from config import CONFIG


def evaluate(model, iterator, optimizer, device, pool):
    model.eval()
    tag_prediction = []
    question = []
    tokenizer = BertTokenizer.from_pretrained("Pretrained_LMs/bert-base-cased2", padding=True, truncation=True)
    text_model = BertModel.from_pretrained("Pretrained_LMs/bert-base-cased2")
    code_model = RobertaModel.from_pretrained("./microsoft/graphcodebert-base1")
    text_model.to(device)
    code_model.to(device)
    with torch.no_grad():
        for _, batch in enumerate(iterator):
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
            # 需要 LongTensor
            input_ids, token_type_ids, labels, attention_mask = input_ids.long(), token_type_ids.long(), labels.long(), attention_mask.long()
            # code_inputs,code_attn_mask,position_idx = code_inputs.long(),code_attn_mask.long(),position_idx.long()
            code_identity = code_identity.long()

            # 梯度清零
            optimizer.zero_grad()
            # 迁移到GPU
            input_ids, token_type_ids, labels, attention_mask = input_ids.to(device), token_type_ids.to(
                device), labels.to(device), attention_mask.to(device)
            code_inputs, code_attn_mask, position_idx = code_inputs.to(device), code_attn_mask.to(
                device), position_idx.to(device)
            code_identity = code_identity.to(device)

            code_token = code_token.long().to(device)

        
            with torch.no_grad():  
                code_token_embed = code_model(code_token)[0]
                text_token_embed = text_model(input_ids)[0]


            code_embeddings = model(code_inputs, code_attn_mask, position_idx)
            # print(len(code_embeddings))
            # code_list = feature.cpu().detach().numpy().tolist() 
            code_list = code_embeddings  

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

            # 获取预测的logits
            loss = output[0]
            prediction_scores = output[1]
            #             print(prediction_scores)
            # decode预测的字
#             '''tag推荐'''
#             for j in range(CONFIG['eval_batch_size']):
#                 
#                 res = ['', '', '', '', '', '', '', '', '', '']
#                 try:
#                     # i 为[mask] 的定位
#                     for i in range(5, 8):

#                        
#                         prediction_tensor, prediction_indices = torch.sort(prediction_scores[j, i], descending=True)

#                         
#                         prediction_index_array = prediction_indices.cpu().detach().numpy()

#                         
#                         for num in range(0, 10):
#                             prediction_index = prediction_index_array[num]
#                             predicted_token = tokenizer.convert_ids_to_tokens([prediction_index])[0]
#                             if i != 7:
#                                 res[num] += predicted_token + ' '
#                             else:
#                                 res[num] += predicted_token

#                     tag_prediction.append(res)
#                 except:
#                     break
#             tokenizer.convert_ids_to_tokens([prediction_index])
            '''question分类'''
            pre_shape = prediction_scores.size()
            for j in range(CONFIG['eval_batch_size']):
             
                res = ['', '', '', '', '', '', '', '', '', '']
                try:
                    
                    for i in range(5, 8):

                        
                        prediction_tensor, prediction_indices = torch.sort(prediction_scores[j, i], descending=True)

                        
                        prediction_index_array = prediction_indices.cpu().detach().numpy()

                       
                        for num in range(0, 10):
                            prediction_index = prediction_index_array[num]
                            predicted_token = tokenizer.convert_ids_to_tokens([prediction_index])[0]
                            if i != 7:
                                res[num] += predicted_token + ' '
                            else:
                                res[num] += predicted_token

                    tag_prediction.append(res)
                except:
                    break
            tokenizer.convert_ids_to_tokens([prediction_index])
            for j in range(pre_shape[0]):
                
                res1 = ""
                
                for i in range(15, 16):
                    
                    prediction_index = torch.argmax(prediction_scores[j, i]).item()
                    predicted_token = tokenizer.convert_ids_to_tokens([prediction_index])[0]
                    res1 += predicted_token
                question.append(res1)

            

    return tag_prediction, question
#     return tag_prediction




