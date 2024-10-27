
import os
import pickle
import torch
import json
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import RobertaTokenizer

from parser import tree_to_token_index, index_to_code_token, remove_comments_and_docstrings
from parser import DFG_python, DFG_java, DFG_ruby, DFG_go, DFG_php, DFG_javascript, DFG_csharp
from parser import (remove_comments_and_docstrings,
                    tree_to_token_index,
                    index_to_code_token,
                    tree_to_variable_index)
from tree_sitter import Language, Parser
from config import CONFIG
dfg_function = {
    'python': DFG_python,
    'java': DFG_java,
    'ruby': DFG_ruby,
    'go': DFG_go,
    'php': DFG_php,
    'javascript': DFG_javascript,
    'c_sharp': DFG_csharp,
    'cpp': DFG_java,
    'c': DFG_python,
}

# load parsers
parsers = {}
for lang in dfg_function:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 code_tokens,
                 code_ids,
                 position_idx,
                 dfg_to_code,
                 dfg_to_dfg,
                 # nl_tokens,
                 # nl_ids,
                 # url,

                 ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.position_idx = position_idx
        self.dfg_to_code = dfg_to_code
        self.dfg_to_dfg = dfg_to_dfg
        # self.nl_tokens = nl_tokens
        # self.nl_ids = nl_ids
        # self.url = url

def extract_dataflow(code: object, parser: object, lang: object) -> object:
    # remove comments
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass
        # obtain dataflow
    if lang == "php":
        code = "<?php" + code + "?>"
    try:
        tree = parser[0].parse(bytes(code, 'utf8'))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code = code.split('\n')
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]
        index_to_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code)
        try:
            DFG, _ = parser[1](root_node, index_to_code, {})
        except:
            DFG = []
        #将tree_sitter解析出来的数据流图中没有数据流流经的节点去掉
        DFG = sorted(DFG, key=lambda x: x[1])
        indexs = set()
        for d in DFG:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG = []
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg = new_DFG
    except:
        dfg = []
    return code_tokens, dfg



def convert_examples_to_features(item):
    code, tokenizer, code_language = item
    # tokenizer = RobertaTokenizer.from_pretrained(CONFIG['tokenizer_name'])
    # code
    if code_language == 'csharp':
        code_language = 'c_sharp'
    parser = parsers[code_language]

    # extract data flow
    code_tokens, dfg = extract_dataflow(code, parser, code_language)
    code_tokens = [tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in
                   enumerate(code_tokens)]
    ori2cur_pos = {}
    ori2cur_pos[-1] = (0, 0)
    for i in range(len(code_tokens)):
        ori2cur_pos[i] = (ori2cur_pos[i - 1][1], ori2cur_pos[i - 1][1] + len(code_tokens[i]))
    code_tokens = [y for x in code_tokens for y in x]
    # truncating
    code_tokens = code_tokens[:CONFIG['code_length'] + CONFIG['data_flow_length'] - 2 - min(len(dfg), CONFIG['data_flow_length'])]
    code_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(code_tokens))]
    dfg = dfg[:CONFIG['code_length'] + CONFIG['data_flow_length'] - len(code_tokens)]
    code_tokens += [x[0] for x in dfg]
    position_idx += [0 for x in dfg]
    code_ids += [tokenizer.unk_token_id for x in dfg]
    padding_length = CONFIG['code_length'] + CONFIG['data_flow_length']- len(code_ids)
    position_idx += [tokenizer.pad_token_id] * padding_length
    code_ids += [tokenizer.pad_token_id] * padding_length
    # reindex
    reverse_index = {}
    for idx, x in enumerate(dfg):
        reverse_index[x[1]] = idx
    for idx, x in enumerate(dfg):
        dfg[idx] = x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)
    dfg_to_dfg = [x[-1] for x in dfg]
    dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
    length = len([tokenizer.cls_token])
    dfg_to_code = [(x[0] + length, x[1] + length) for x in dfg_to_code]
    # # nl
    # nl = ' '.join(js['docstring_tokens'])
    # nl_tokens = tokenizer.tokenize(nl)[:CONFIG['nl_length'] - 2]
    # nl_tokens = [tokenizer.cls_token] + nl_tokens + [tokenizer.sep_token]
    # nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    # padding_length = CONFIG['nl_length'] - len(nl_ids)
    # nl_ids += [tokenizer.pad_token_id] * padding_length

    return code_tokens, code_ids, position_idx, dfg_to_code, dfg_to_dfg
    # return InputFeatures(code_tokens, code_ids, position_idx, dfg_to_code, dfg_to_dfg)#, nl_tokens, nl_ids, js['url']

def  getcodeAttenMask(position_idx,code_ids,dfg_to_code,dfg_to_dfg):
    attn_mask = np.zeros((CONFIG['code_length'] + CONFIG['data_flow_length'],
                          CONFIG['code_length'] + CONFIG['data_flow_length']), dtype=np.bool)
    # calculate begin index of node and max length of input
    node_index = sum([i > 1 for i in position_idx])
    max_length = sum([i != 1 for i in position_idx])
    # sequence can attend to sequence
    attn_mask[:node_index, :node_index] = True
    # special tokens attend to all tokens
    for idx, i in enumerate(code_ids):
        if i in [0, 2]:
            attn_mask[idx, :max_length] = True
    # nodes attend to code tokens that are identified from
    for idx, (a, b) in enumerate(dfg_to_code):
        if a < node_index and b < node_index:
            attn_mask[idx + node_index, a:b] = True
            attn_mask[a:b, idx + node_index] = True
    # nodes attend to adjacent nodes
    for idx, nodes in enumerate(dfg_to_dfg):
        for a in nodes:
            if a + node_index < len(position_idx):
                attn_mask[idx + node_index, a + node_index] = True
    return attn_mask


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None, pool=None):
        self.args = args
        prefix = file_path.split('/')[-1][:-6]
        cache_file = args.output_dir + '/' + prefix + '.pkl'
        if os.path.exists(cache_file):
            self.examples = pickle.load(open(cache_file, 'rb'))
        else:
            self.examples = []
            data = []
            with open(file_path) as f:
                for line in f:
                    line = line.strip()
                    js = json.loads(line)
                    data.append((js, tokenizer, args))
            self.examples = pool.map(convert_examples_to_features, tqdm(data, total=len(data)))
            pickle.dump(self.examples, open(cache_file, 'wb'))

        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
               print("*** Example ***")
               print("idx: {}".format(idx))
               print("code_tokens: {}".format([x.replace('\u0120', '_') for x in example.code_tokens]))
               print("code_ids: {}".format(' '.join(map(str, example.code_ids))))
               print("position_idx: {}".format(example.position_idx))
               print("dfg_to_code: {}".format(' '.join(map(str, example.dfg_to_code))))
               print("dfg_to_dfg: {}".format(' '.join(map(str, example.dfg_to_dfg))))
               print("nl_tokens: {}".format([x.replace('\u0120', '_') for x in example.nl_tokens]))
               print("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        # calculate graph-guided masked function
        attn_mask = np.zeros((self.args.code_length + self.args.data_flow_length,
                              self.args.code_length + self.args.data_flow_length), dtype=np.bool)
        # calculate begin index of node and max length of input
        node_index = sum([i > 1 for i in self.examples[item].position_idx])
        max_length = sum([i != 1 for i in self.examples[item].position_idx])
        # sequence can attend to sequence
        attn_mask[:node_index, :node_index] = True
        # special tokens attend to all tokens
        for idx, i in enumerate(self.examples[item].code_ids):
            if i in [0, 2]:
                attn_mask[idx, :max_length] = True
        # nodes attend to code tokens that are identified from
        for idx, (a, b) in enumerate(self.examples[item].dfg_to_code):
            if a < node_index and b < node_index:
                attn_mask[idx + node_index, a:b] = True
                attn_mask[a:b, idx + node_index] = True
        # nodes attend to adjacent nodes
        for idx, nodes in enumerate(self.examples[item].dfg_to_dfg):
            for a in nodes:
                if a + node_index < len(self.examples[item].position_idx):
                    attn_mask[idx + node_index, a + node_index] = True

        return (torch.tensor(self.examples[item].code_ids),
                torch.tensor(attn_mask),
                torch.tensor(self.examples[item].position_idx),
                torch.tensor(self.examples[item].nl_ids))
