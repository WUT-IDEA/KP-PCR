import torch
from transformers import BertTokenizer, BertForMaskedLM, AdamW, BertModel
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel

from myDataset import convert_examples_to_features, getcodeAttenMask
from config import CONFIG



def tokenize_function(sample):

    tokenizer = BertTokenizer.from_pretrained("./Pretrained_LMs/bert-base-cased2", padding=True, truncation=True)
    tokenized = tokenizer(
        sample['0'],  
        truncation=True,
        padding="max_length",
        max_length=CONFIG['nl_length'],
        return_special_tokens_mask=True,
    )
    

    tokenizer1 = RobertaTokenizer.from_pretrained("./microsoft/graphcodebert-base1")
    code_tokenized = tokenizer1(sample['0'],  truncation=True, padding="max_length", max_length=CONFIG['nl_length'], return_special_tokens_mask=True)

    # codedata=sample['2']
    #
    #
    code_tokens_list = []
    code_ids_list = []
    position_idx_list = []
    dfg_to_code_list = []
    dfg_to_dfg_list = []
    code_attn_mask_list = []
    for i in range(len(sample['2'])):

        code = sample['2'][i]
        code_language = sample['3'][i]
        # code='func getAllDepTypes() []string {\n\tdepTypes := make([]string, 0, len(cmds))\n\tfor depType := range cmds {\n\t\tdepTypes = append(depTypes, depType)\n\t}\n\tsort.Strings(depTypes)\n\treturn depTypes\n}'
        # lang=sample['4'][i]
        # maxlen_c = 120
        try:
            code_tokens, code_ids, position_idx, dfg_to_code, dfg_to_dfg = convert_examples_to_features(
                (code, tokenizer1, code_language))
            code_attn_mask = getcodeAttenMask(position_idx, code_ids, dfg_to_code, dfg_to_dfg)
            code_tokens_list.append(code_tokens)
            code_ids = [0 if x == None else x for x in code_ids]
            code_ids_list.append(code_ids)
            position_idx_list.append(position_idx)

            code_attn_mask_list.append(code_attn_mask)
            # tokens_tensor = tokenizer1(
            #                 code,
            #                 add_special_tokens=False,  
            #                 max_length = maxlen_c,      
            #                 pad_to_max_length=True, 
            #                 return_tensors='pt'      
            #            ) 
        except:
            code_tokens, code_ids, position_idx, dfg_to_code, dfg_to_dfg = convert_examples_to_features(
                ('null', tokenizer1,  code_language))
            # code_tokens_list.append(code_tokens)
            code_attn_mask = getcodeAttenMask(position_idx, code_ids, dfg_to_code, dfg_to_dfg)
            code_ids = [0 if x == None else x for x in code_ids]
            code_ids_list.append(code_ids)
            position_idx_list.append(position_idx)
            code_attn_mask_list.append(code_attn_mask)

            
            
            
    tokenized['code_ids'] = code_ids_list

    tokenized['position_idx'] = position_idx_list
    tokenized['code_attn_mask'] = code_attn_mask_list
    tokenized['code_identity'] = sample['4']

    tokenized['code_token'] = code_tokenized['input_ids']
    # tokenized['dfg_to_code'] = dfg_to_code_list
    # tokenized['dfg_to_dfg'] = dfg_to_dfg_list
    return tokenized




# def tokenize_function(sample):

#     tokenizer = BertTokenizer.from_pretrained("./Pretrained_LMs/bert-base-cased2", padding=True, truncation=True)
#     tokenized = tokenizer(
#         sample['0'],  # 这里是传入的列
#         truncation=True,
#         padding="max_length",
#         max_length=CONFIG['nl_length'],
#         return_special_tokens_mask=True,
#     )

    
#     wiki_tokenized = tokenizer(
#         sample['6'],  
#         truncation=True,
#         padding="max_length",
#         max_length=80,
#         return_special_tokens_mask=True,
#     )
    

#     tokenizer1 = RobertaTokenizer.from_pretrained("./microsoft/graphcodebert-base1")
#     code_tokenized = tokenizer1(sample['0'],  truncation=True, padding="max_length", max_length=CONFIG['nl_length'], return_special_tokens_mask=True)

#     # codedata=sample['2']
#     #
#     #
#     code_tokens_list = []
#     code_ids_list = []
#     position_idx_list = []
#     dfg_to_code_list = []
#     dfg_to_dfg_list = []
#     code_attn_mask_list = []
#     for i in range(len(sample['2'])):

#         code = sample['2'][i]
#         code_language = sample['3'][i]
#         # code='func getAllDepTypes() []string {\n\tdepTypes := make([]string, 0, len(cmds))\n\tfor depType := range cmds {\n\t\tdepTypes = append(depTypes, depType)\n\t}\n\tsort.Strings(depTypes)\n\treturn depTypes\n}'

#         # lang=sample['4'][i]
#         # maxlen_c = 120
#         try:
#             code_tokens, code_ids, position_idx, dfg_to_code, dfg_to_dfg = convert_examples_to_features(
#                 (code, tokenizer1, code_language))
#             code_attn_mask = getcodeAttenMask(position_idx, code_ids, dfg_to_code, dfg_to_dfg)
#             code_tokens_list.append(code_tokens)
#             code_ids = [0 if x == None else x for x in code_ids]
#             code_ids_list.append(code_ids)
#             position_idx_list.append(position_idx)

#             code_attn_mask_list.append(code_attn_mask)
#             # tokens_tensor = tokenizer1(
#             #                 code,
#             #                 add_special_tokens=False, 
#             #                 max_length = maxlen_c,          
#             #                 pad_to_max_length=True,  
#             #                 return_tensors='pt'      
#             #            ) 
#         except:
#             code_tokens, code_ids, position_idx, dfg_to_code, dfg_to_dfg = convert_examples_to_features(
#                 ('null', tokenizer1,  code_language))
#             # code_tokens_list.append(code_tokens)
#             code_attn_mask = getcodeAttenMask(position_idx, code_ids, dfg_to_code, dfg_to_dfg)
#             code_ids = [0 if x == None else x for x in code_ids]
#             code_ids_list.append(code_ids)
#             position_idx_list.append(position_idx)
#             code_attn_mask_list.append(code_attn_mask)

            
            
#     tokenized['code_ids'] = code_ids_list

#     tokenized['position_idx'] = position_idx_list
#     tokenized['code_attn_mask'] = code_attn_mask_list
#     tokenized['code_identity'] = sample['4']

#     tokenized['code_token'] = code_tokenized['input_ids']
#     tokenized['wiki_input_ids'] = wiki_tokenized['input_ids'] 
#     # tokenized['dfg_to_code'] = dfg_to_code_list
#     # tokenized['dfg_to_dfg'] = dfg_to_dfg_list
#     return tokenized