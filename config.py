CONFIG = {

    'cuda': 'cuda:1', 
    
    'lang': 'go',
    'model_name_or_path': './microsoft/graphcodebert-base1',
    'config_name': './microsoft/graphcodebert-base1',
    'tokenizer_name': './microsoft/graphcodebert-base1',

    'epoch': 6,
#     训练
#     'do_train': True,
#     'do_eval': False,   
#     测试
    'do_train': False,
    'do_eval': True,
    

    'train_model_path': 'Train_Model/wuguanci/epoch_',
    'dev_model_path': 'Train_Model/question01/epoch_1',
    
    'train_data_file': '../data_codereview/data/GraphCodeBert_data/question_train1_set_01_wuguanci.csv',
    'dev_data_file': '../data_codereview/data/GraphCodeBert_data/question_dev1_set_01_des.csv',
    
    
    'start_index': 5,
    'end_index': 8,
    'start_question_index': 15,
    'end_question_index': 16,
    
    'nl_length': 330,
    'code_length': 150,
    'maxlen_pre_concat': 250,
    'maxlen_text_concat': 250,
    'maxlen_code_concat': 300,
    'data_flow_length': 64,

    'train_batch_size': 4,
    'eval_batch_size': 4,
    'learning_rate': 1e-5,
    'max_grad_norm': 1.0,
    'num_train_epochs': 5,
    'seed': 123456,
}
