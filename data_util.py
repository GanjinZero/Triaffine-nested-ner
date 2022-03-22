import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import ujson
import numpy as np
import torch
from transformers import AutoTokenizer
import os
from torch.utils.data import Dataset
from span_utils import iou
import h5py


class NestedNERDataset(Dataset):
    def __init__(self, version, mode, tokenizer, truncate_length=128, schema="span", use_context=True,
                 token_schema="BIES", soft_iou=0.7, bert_embed=""):
        self.version = version
        self.mode = mode
        self.schema = schema
        # if isinstance(tokenizer, str):
        #     self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        # else:
        #     self.tokenizer = tokenizer
        
        self.tokenizer_list = []
        for t in tokenizer.split(','):
            self.tokenizer_list.append(AutoTokenizer.from_pretrained(t))
        
        self.file_path = self._get_file_path(self.version, self.mode)
        self.df = self._load_file(self.file_path)
        self.type2id = self._get_type(self.version)
        self.id2type = {idx: tp for tp, idx in self.type2id.items()}

        self.truncate_length = truncate_length
        self.use_context = use_context

        self.word2id, self.id2word = self._load_vocab(self.version)
        self.pos2id = self._load_pos(self.version)
        self.char2id = self._load_char(self.version)

        # hard code
        # self.max_word_count = max(160, self.truncate_length)
        self.max_word_count = 200
        self.max_char_count = 70
        self.max_entity_count = 30
        
        self.token_schema = token_schema
        assert token_schema in ['BE', 'BIE', 'BIES',
                                'BE-type', 'BIE-type', 'BIES-type']
        if self.token_schema == "BE":
            self.token_label_count = 2
        elif self.token_schema == "BIE":
            self.token_label_count = 3
        elif self.token_schema == "BIES":
            self.token_label_count = 4
        elif self.token_schema == "BE-type":
            self.token_label_count = 2 * len(self.type2id)
        elif self.token_schema == "BIE-type":
            self.token_label_count = 3 * len(self.type2id)
        elif self.token_schema == "BIES-type":
            self.token_label_count = 4 * len(self.type2id)
            
        self.soft_iou = soft_iou
        
        self.bert_embed_path = bert_embed
        if self.bert_embed_path:
            print(self.bert_embed_path)
        #    self.examples = h5py.File(self.bert_embed_path, 'r', libver='latest')

    def open_hdf5(self):
        self.examples = h5py.File(self.bert_embed_path, 'r', libver='latest')

    def _get_file_path(self, version, mode):
        #return f"data/{version}/{mode}_sample.json"
        if version.startswith("ace"):
            return f"data/{version}/{version}_{mode}_context.json"
        elif version == "genia91":
            if mode == "test":
                return f"data/{version}/genia_test_context.json"
            else:
                return f"data/{version}/genia_train_dev_context.json"
        elif version == "kbp":
            return f"data/{version}/{mode}_context.json"

    def _load_file(self, file_path):
        with open(file_path, "r") as f:
            df = ujson.load(f)
        self.len = len(df)
        return df

    def __len__(self):
        return self.len

    def _get_type(self, version):
        if version.find("genia") >= 0:
            type_list = ["protein", "cell_type", "cell_line", "DNA", "RNA"]
        if version.find('ace') >= 0:
            type_list = ['PER', 'LOC', 'ORG', 'GPE', 'FAC', 'VEH', 'WEA']
        if version.find('kbp') >= 0:
            type_list = ['GPE', 'FAC', 'ORG', 'PER', 'LOC']
        return {tp: idx for idx, tp in enumerate(type_list)}

    def _load_vocab(self, version):
        if version.find('v2') >= 0 or version.find('v3') >= 0 or version.find('v4') >= 0:
            version = version[:-3]
        vocab_path = f'data/{version}/word2id.json'
        with open(vocab_path, "r") as f:
            word2id = ujson.load(f)
        id2word = {id: word for word, id in word2id.items()}
        return word2id, id2word

    def _load_pos(self, version):
        if version.find('v2') >= 0 or version.find('v3') >= 0 or version.find('v4') >= 0:
            version = version[:-3]
        pos_path = f'data/{version}/pos2id.json'
        with open(pos_path, "r") as f:
            pos2id = ujson.load(f)
        return pos2id

    def _load_char(self, version):
        if version.find('v2') >= 0 or version.find('v3') >= 0 or version.find('v4') >= 0:
            version = version[:-3]
        char_path = f'data/{version}/char2id.json'
        with open(char_path, "r") as f:
            char2id = ujson.load(f)
        return char2id

    def pad(self, l, pad_token_length, pad_token_id, reverse=False):
        if len(l) > pad_token_length:
            if not reverse:
                return l[0:pad_token_length]
            return l[-pad_token_length:]
        if not reverse:
            return l + [pad_token_id] * (pad_token_length - len(l))
        return [pad_token_id] * (pad_token_length - len(l)) + l
    
    def shift(self, l, shift_idx):
        return [x + shift_idx for x in l]

    def bert_tokenize(self, tokenizer, tokens, ltokens=None, rtokens=None):
        l_tokenized = []
        l_length = 0
        l_context_ce_mask = []
        l_context_subword_group = []
        if ltokens is not None:
            for idx, word in enumerate(ltokens):
                subword = tokenizer.tokenize(word)
                l_tokenized.extend(subword)
                if len(subword) >= 1:
                    l_context_ce_mask.append(1)
                    l_context_ce_mask.extend([0] * (len(subword) - 1))
                    l_context_subword_group.extend([idx] * len(subword))
            l_length = len(l_tokenized)
        l_ce_mask = [0] * l_length
        l_subword_group = [-1] * l_length
        
        r_tokenized = []
        r_length = 0
        r_context_ce_mask = []
        r_context_subword_group = []
        if rtokens is not None:
            for idx, word in enumerate(rtokens):
                subword = tokenizer.tokenize(word)
                r_tokenized.extend(subword)
                if len(subword) >= 1:
                    r_context_ce_mask.append(1)
                    r_context_ce_mask.extend([0] * (len(subword) - 1))
                    r_context_subword_group.extend([idx] * len(subword))
            r_length = len(r_tokenized)
        r_ce_mask = [0] * r_length
        r_subword_group = [-1] * r_length

        tokenized = []
        ce_mask = []
        subword_group = []
        for idx, word in enumerate(tokens):
            subword = tokenizer.tokenize(word)
            tokenized.extend(subword)
            #assert len(subword) >= 1
            if len(subword) >= 1:
                ce_mask.append(1)
                ce_mask.extend([0] * (len(subword) - 1))
                subword_group.extend([idx] * len(subword))
            else:
                tokenized.extend(['[UNK]'])
                ce_mask.append(1)
                subword_group.extend([idx])

        if ltokens is None and rtokens is None:
            all_tokens = ['[CLS]'] + tokenized + ['[SEP]']
            all_ce_mask = [0] + ce_mask + [0]
            all_subword_group = [-1] + subword_group + [-1]
            token_type_ids = [0] + [0] * len(tokenized) + [0]
            context_ce_mask = [0] + ce_mask + [0]
            context_subword_group = [-1] + subword_group + [-1]
        if ltokens is None and rtokens is not None:
            all_tokens = ['[CLS]'] + tokenized + \
                ['[SEP]'] + r_tokenized + ['[SEP]']
            all_ce_mask = [0] + ce_mask + [0] + r_ce_mask + [0]
            all_subword_group = [-1] + subword_group + [-1] + r_subword_group + [-1]
            token_type_ids = [0] + [0] * len(tokenized) + \
                [0] + [1] * len(r_tokenized) + [1]
            context_ce_mask = [0] + ce_mask + [0] + r_context_ce_mask + [0]
            context_subword_group = [-1] + subword_group + [-1] + \
                self.shift(r_context_subword_group, max(subword_group) + 1) + [-1]
        if ltokens is not None and rtokens is None:
            all_tokens = ['[CLS]'] + l_tokenized + \
                ['[SEP]'] + tokenized + ['[SEP]']
            all_ce_mask = [0] + l_ce_mask + [0] + ce_mask + [0]
            all_subword_group = [-1] + l_subword_group + [-1] + subword_group + [-1]
            token_type_ids = [1] + [1] * len(l_tokenized) + \
                [0] + [0] * len(tokenized) + [0]
            context_ce_mask = [0] + l_context_ce_mask + [0] + ce_mask + [0]
            context_subword_group = [-1] + l_context_subword_group + [-1] + \
                self.shift(subword_group, max(l_context_subword_group) + 1) + [-1]
        if ltokens is not None and rtokens is not None:
            all_tokens = ['[CLS]'] + l_tokenized + ['[SEP]'] + \
                tokenized + ['[SEP]'] + r_tokenized + ['[SEP]']
            all_ce_mask = [0] + l_ce_mask + [0] + \
                ce_mask + [0] + r_ce_mask + [0]
            all_subword_group = [-1] + l_subword_group + [-1] + \
                subword_group + [-1] + r_subword_group + [-1]
            token_type_ids = [1] + [1] * len(l_tokenized) + [0] + \
                [0] * len(tokenized) + [0] + [1] * len(r_tokenized) + [1]
            context_ce_mask = [0] + l_context_ce_mask + [0] + ce_mask + [0] + r_context_ce_mask + [0]
            context_subword_group = [-1] + l_context_subword_group + [-1] + \
                self.shift(subword_group, max(l_context_subword_group) + 1) + [-1] + \
                self.shift(r_context_subword_group, max(l_context_subword_group) + max(subword_group) + 2) + [-1]

        no_context_tokens = ['[CLS]'] + tokenized + ['[SEP]']
        no_context_ce_mask = [0] + ce_mask + [0]
        no_context_subword_group = [-1] + subword_group + [-1]
        no_context_token_type_ids = [0] + [0] * len(tokenized) + [0]

        input_ids = tokenizer.convert_tokens_to_ids(all_tokens)
        attention_mask = [1] * len(input_ids)

        no_context_input_ids = tokenizer.convert_tokens_to_ids(
            no_context_tokens)
        no_context_attention_mask = [1] * len(no_context_input_ids)

        # truncation
        if len(input_ids) <= self.truncate_length:
            input_ids = self.pad(
                input_ids, self.truncate_length, tokenizer.pad_token_id)
            attention_mask = self.pad(attention_mask, self.truncate_length, 0)
            all_ce_mask = self.pad(all_ce_mask, self.truncate_length, 0)
            all_subword_group = self.pad(all_subword_group, self.truncate_length, -1)
            token_type_ids = self.pad(token_type_ids, self.truncate_length, 0)
            context_ce_mask = self.pad(context_ce_mask, self.truncate_length, 0)
            context_subword_group = self.pad(context_subword_group, self.truncate_length, -1)
            # print(all_tokens)
        else:
            if len(no_context_input_ids) > self.truncate_length:
                input_ids = no_context_input_ids[0:self.truncate_length]
                attention_mask = no_context_attention_mask[0:self.truncate_length]
                all_ce_mask = no_context_ce_mask[0:self.truncate_length]
                all_subword_group = no_context_subword_group[0:self.truncate_length]
                token_type_ids = no_context_token_type_ids[0:self.truncate_length]
                context_ce_mask = all_ce_mask
                context_subword_group = all_subword_group
            else:
                if ltokens is None:
                    # tokens + part_rtokens
                    start_idx = 0
                    end_idx = self.truncate_length
                elif rtokens is None:
                    # part_ltokens + tokens
                    end_idx = len(input_ids)
                    start_idx = end_idx - self.truncate_length
                else:
                    # part_ltokens + tokens + part_rtokens
                    # first [sep] indicates tokens start idx
                    tokens_start_idx = all_tokens.index('[SEP]')
                    tokens_end_idx = tokens_start_idx + \
                        len(no_context_input_ids)
                    context_len = (self.truncate_length -
                                   (tokens_end_idx - tokens_start_idx)) // 2
                    start_idx = max(0, tokens_start_idx - context_len)
                    end_idx = start_idx + self.truncate_length
                    if end_idx >= len(input_ids):
                        end_idx = len(input_ids)
                        start_idx = end_idx - self.truncate_length
                input_ids = input_ids[start_idx:end_idx]
                attention_mask = attention_mask[start_idx:end_idx]
                all_ce_mask = all_ce_mask[start_idx:end_idx]
                all_subword_group = all_subword_group[start_idx:end_idx]
                token_type_ids = token_type_ids[start_idx:end_idx]
                context_ce_mask = context_ce_mask[start_idx:end_idx]
                context_subword_group = context_subword_group[start_idx:end_idx]
                
        min_context_subword_group = min([num for num in context_subword_group if num >= 0])
        context_subword_group = [num - min_context_subword_group if num >= 0 else num for num in context_subword_group]
        #print(context_subword_group)

        return input_ids, attention_mask, all_ce_mask, token_type_ids, all_subword_group, \
            context_ce_mask, context_subword_group

    def word_tokenize(self, tokens, reverse=False):
        if tokens is None:
            return self.pad([], self.max_word_count, self.word2id['[PAD]']), 0
        input_word = [self.word2id.get(
            token, self.word2id['[UNK]']) for token in tokens]
        word_count = min(len(input_word), self.max_word_count)
        return self.pad(input_word, self.max_word_count, self.word2id['[PAD]'], reverse), word_count

    def char_tokenize(self, tokens, reverse=False):
        if tokens is None:
            input_char = []
        else:
            input_char = [self.pad([self.char2id.get(char, self.char2id['[UNK]']) for char in token], self.max_char_count, self.char2id['[PAD]']) for token in tokens]
        return self.pad(input_char,
                        self.max_word_count,
                        [self.char2id['[PAD]']] * self.max_char_count,
                        reverse)

    def pos_tokenize(self, pos, reverse=False):
        if pos is None:
            input_pos = []
        else:
            input_pos = [self.pos2id.get(p, self.pos2id['[UNK]']) for p in pos]
        return self.pad(input_pos, self.max_word_count, self.pos2id['[PAD]'], reverse)

    def convert(self, word_count, entities):
        # span
        if self.schema in ['span']:
            new_label = np.full(
                (self.max_word_count, self.max_word_count), -100, dtype='int64')
            new_label[0:word_count,0:word_count] += \
                np.triu((100 + len(self.type2id)) * np.ones((word_count, word_count), dtype='int64'))
            for ent in entities:
                if ent['end'] - 1 < self.max_word_count:
                    new_label[ent['start'], ent['end']-1] = self.type2id[ent['type']]
            return new_label.tolist(), None, None, None

        # seq2seq
        if self.schema in ['DETRSeq', 'DETR']:
            label = []
            for ent in entities:
                label.append([ent['start'], ent['end']-1, self.type2id[ent['type']]])
            label = sorted(label, key=lambda x: (x[0], x[1], x[2]))
            if self.schema == 'DETRSeq':
                label = label + \
                    [[-100, -100, len(self.type2id)]] + \
                    [[-100, -100, -100]] * (self.max_entity_count - 1 - len(label))
            if self.schema == "DETR":
                label = label + \
                    [[-100, -100, -100]] * (self.max_entity_count - len(label))
            return label, None, None, None
        
        if self.schema == "softspan":
            new_label = np.full(
                (self.max_word_count, self.max_word_count), -100, dtype='int64')
            ori_label = np.full(
                (self.max_word_count, self.max_word_count), -100, dtype='int64')
            head_label = np.full(
                (self.max_word_count, self.max_word_count), -100, dtype='int64')
            tail_label = np.full(
                (self.max_word_count, self.max_word_count), -100, dtype='int64')
            
            new_label[0:word_count,0:word_count] += \
                np.triu((100 + len(self.type2id)) * np.ones((word_count, word_count), dtype='int64'))
            ori_label[0:word_count,0:word_count] += \
                np.triu((100 + len(self.type2id)) * np.ones((word_count, word_count), dtype='int64'))
            tail_label[0:word_count,0:word_count] = np.arange(word_count).reshape(1, -1).repeat(word_count, 0)
            head_label[0:word_count,0:word_count] = tail_label[0:word_count,0:word_count].transpose()
            iou_score = np.zeros((self.max_word_count, self.max_word_count)) # record max iou score between this span with any entity
            
            for ent in entities:
                st = ent['start']
                ed = ent['end'] - 1
                tp = self.type2id[ent['type']]
                new_label[st, ed] = tp
                ori_label[st, ed] = tp
                
                for i in range(0, ed + 1):
                    for j in range(i, word_count):
                        iou_now = iou((i, j), (st, ed))
                        if iou_now > max(self.soft_iou, iou_score[i,j]):
                            iou_score[i, j] = iou_now
                            new_label[i, j] = tp
                            head_label[i, j] = st
                            tail_label[i, j] = ed
            
            head_label[new_label == len(self.type2id)] = -100
            head_label[new_label == -100] = -100
            tail_label[new_label == len(self.type2id)] = -100
            tail_label[new_label == -100] = -100
            
            return new_label.tolist(), head_label.tolist(), tail_label.tolist(), ori_label.tolist()
        
    def convert_token(self, word_count, entities):
        #label = np.full((self.max_word_count, self.token_label_count), -100, dtype='int64')
        label = np.full((1000, self.token_label_count), -100, dtype='int64')

        label[0:word_count] = 0
        for ent in entities:
            st = ent['start']
            ed = ent['end']-1
            tp = self.type2id[ent['type']]
            if self.token_schema == "BE":
                label[st][0] = 1
                label[ed][1] = 1
            elif self.token_schema == "BIE":
                # BEI exactly
                label[st][0] = 1
                label[ed][1] = 1
                for i in range(st + 1, ed):
                    label[i][2] = 1   
            elif self.token_schema == "BIES":
                if st == ed:
                    label[st][3] = 1
                else:
                    label[st][0] = 1
                    label[ed][1] = 1
                    for i in range(st + 1, ed):
                        label[i][2] = 1
            elif self.token_schema == "BE-type":
                label[st][2 * tp] = 1
                label[ed][2 * tp + 1] = 1
            elif self.token_schema == "BIE-type":
                # BEI exactly
                label[st][3 * tp] = 1
                label[ed][3 * tp + 1] = 1
                for i in range(st + 1, ed):
                    label[i][3 * tp + 2] = 1   
            elif self.token_schema == "BIES-type":
                if st == ed:
                    label[st][4 * tp + 3] = 1
                else:
                    label[st][4 * tp] = 1
                    label[ed][4 * tp + 1] = 1
                    for i in range(st + 1, ed):
                        label[i][4 * tp + 2] = 1
                            
        return label[0:self.max_word_count].tolist()
    
    def convert_trans(self, word_count, entities):
        head_trans_label = np.full(
            (self.max_word_count, self.max_word_count), -100, dtype='int64')
        head_trans_label[0:word_count,0:word_count] = 0
        tail_trans_label = np.full(
            (self.max_word_count, self.max_word_count), -100, dtype='int64')
        tail_trans_label[0:word_count,0:word_count] = 0
        
        start_list = []
        end_list = []
        for ent in entities:
            start_list.append(ent['start'])
            end_list.append(ent['end'] - 1)
        
        for i in range(len(start_list)):
            smallest_start = -1
            smallest_end = 200
            for j in range(len(start_list)):
                if i == j:
                    continue
                if start_list[j] <= start_list[i] and end_list[i] <= end_list[j] and end_list[j] - start_list[j] < smallest_end - smallest_start:
                    smallest_start = start_list[j]
                    smallest_end = end_list[j]
            if smallest_start >= 0:
                if 0 <= start_list[i] < self.max_word_count and 0 <= smallest_start < self.max_word_count:
                    head_trans_label[start_list[i], smallest_start] = 1
                if 0 <= end_list[i] < self.max_word_count and 0 <= smallest_end < self.max_word_count:
                    tail_trans_label[end_list[i], smallest_end] = 1
            
        return head_trans_label.tolist(), tail_trans_label.tolist()
    
    def group_unsqueeze(self, subword_group):
        # list of list
        res = []
        for grp in subword_group:
            tmp = []
            grp = np.array(grp)
            max_idx = max(grp)
            for i in range(max_idx + 1):
                tmp.append((grp==i).tolist())
            tmp = self.pad(tmp, self.max_word_count, (np.array([False] * self.truncate_length).tolist()))
            res.append(tmp)
        return res

    def __getitem__(self, index):
        line = self.df[index]
        tokens = line['tokens']
        if self.use_context:
            ltokens = line['ltokens']
            if not ltokens:
                ltokens = None
            rtokens = line['rtokens']
            if not rtokens:
                rtokens = None
        else:
            ltokens = None
            rtokens = None
        pos = line.get('pos', None)

        input_ids = []
        attention_mask = []
        ce_mask = []
        token_type_ids = []
        subword_group = []
        context_ce_mask = []
        context_subword_group = []

        for t in self.tokenizer_list:
            input_ids_0, attention_mask_0, ce_mask_0, token_type_ids_0, subword_group_0, \
                context_ce_mask_0, context_subword_group_0 = \
                self.bert_tokenize(t, tokens, ltokens, rtokens)
            input_ids.append(input_ids_0)
            attention_mask.append(attention_mask_0)
            ce_mask.append(ce_mask_0)
            token_type_ids.append(token_type_ids_0)
            subword_group.append(subword_group_0)
            context_ce_mask.append(context_ce_mask_0)
            context_subword_group.append(context_subword_group_0)

        max_tok_len = max([len(input_ids_0) for input_ids_0 in input_ids])
        input_ids = [self.pad(input_ids_0, max_tok_len, self.tokenizer_list[idx].pad_token_id) for idx, input_ids_0 in enumerate(input_ids)]
        attention_mask = [self.pad(att, max_tok_len, 0) for att in attention_mask]
        ce_mask = [self.pad(att, max_tok_len, 0) for att in ce_mask]
        token_type_ids = [self.pad(att, max_tok_len, 0) for att in token_type_ids]
        subword_group = [self.pad(grp, max_tok_len, -1) for grp in subword_group]
        context_ce_mask = [self.pad(att, max_tok_len, 0) for att in context_ce_mask]
        context_subword_group = [self.pad(grp, max_tok_len, -1) for grp in context_subword_group]
        
        # import ipdb; ipdb.set_trace()
        
        subword_group = self.group_unsqueeze(subword_group)
        context_subword_group = self.group_unsqueeze(context_subword_group)
        
        context_map = []
        for idx, _ in enumerate(self.tokenizer_list):
            tmp = []
            count = 0
            for i, m in enumerate(context_ce_mask[idx]):
                if m == 1:
                    if ce_mask[idx][i] == 1:
                        tmp.append(count)
                    count += 1
            context_map.append(tmp)
        context_map = [self.pad(att, max_tok_len, 0) for att in context_map]
                
        input_word, word_count = self.word_tokenize(tokens)
        input_char = self.char_tokenize(tokens)
        
        l_input_word, _ = self.word_tokenize(ltokens, True)
        r_input_word, _ = self.word_tokenize(rtokens)
        l_input_char = self.char_tokenize(ltokens, True)
        r_input_char = self.char_tokenize(rtokens)
        
        input_pos = self.pos_tokenize(pos)
        l_input_pos = self.pos_tokenize(line.get('lpos', []), True)
        r_input_pos = self.pos_tokenize(line.get('rpos', []))

        # deal with label
        entities = line['entities']
        span_label, head_label, tail_label, ori_label = self.convert(word_count, entities)
        token_label = self.convert_token(word_count, entities)
        head_trans_label, tail_trans_label = self.convert_trans(word_count, entities)
        
        # hard code
        if self.bert_embed_path:
            if not hasattr(self, 'examples'):
                self.open_hdf5()
            if h5py.__version__.startswith('2'):
                bert_embed = self.examples[str(index)].value
            elif h5py.__version__.startswith('3'):
                bert_embed = self.examples[str(index)][()]
        else:
            # hard code
            bert_embed = input_ids

        if head_label is None:
            head_label = input_ids
            tail_label = input_ids # just padding
            ori_label = input_ids
            
        return input_ids, attention_mask, ce_mask, token_type_ids, subword_group, \
               context_ce_mask, context_subword_group, context_map, \
               input_word, input_char, input_pos, \
               l_input_word, l_input_char, l_input_pos, \
               r_input_word, r_input_char, r_input_pos, \
               span_label, token_label, head_label, tail_label, ori_label, \
               head_trans_label, tail_trans_label, bert_embed


def my_collate_fn(batch):
    type_count = len(batch[0])
    batch_size = len(batch)
    output = ()
    for i in range(type_count):
        tmp = []
        for item in batch:
            tmp.extend(item[i])
        if len(tmp) <= batch_size:
            output += (torch.LongTensor(tmp),)
        elif isinstance(tmp[0], int):
            output += (torch.LongTensor(tmp).reshape(batch_size, -1),)
        elif isinstance(tmp[0], list):
            dim_y = len(tmp[0])
            if isinstance(tmp[0][0], int):
                output += (torch.LongTensor(tmp).reshape(batch_size, -1, dim_y),)
            elif isinstance(tmp[0][0], float):
                output += (torch.FloatTensor(tmp).reshape(batch_size, -1, dim_y),)
            elif isinstance(tmp[0][0], list):
                dim_z = len(tmp[0][0])
                output += (torch.FloatTensor(tmp).reshape(batch_size, -1, dim_y, dim_z), )
            elif isinstance(tmp[0][0], np.ndarray):
                dim_z = tmp[0][0].shape[-1]
                output += (torch.FloatTensor(tmp).reshape(batch_size, -1, dim_y, dim_z), )
        elif isinstance(tmp[0], np.ndarray):
            dim_y = tmp[0].shape[-1]
            output += (torch.FloatTensor(tmp).reshape(batch_size, -1, dim_y),)
    return output


def get_cls_num_list(version):
    mode = 'train'
    if version == "ace04":
        file_path = f"data/ace04/ace04_{mode}_context.json"
    if version == "ace05":
        file_path = f"data/ace05/ace05_{mode}_context.json"
    if version == "genia91":
        if mode == "train":
            file_path = "data/genia91/genia_train_dev_context.json"
        else:
            file_path = "data/genia91/genia_test_context.json"
    if version == "kbp":
        file_path = f"data/kbp/{mode}_context.json"
        
    with open(file_path, "r") as f:
        df = ujson.load(f)
        
    if version.find("genia") >= 0:
        type_list = ["protein", "cell_type", "cell_line", "DNA", "RNA"]
    if version.find('ace') >= 0:
        type_list = ['PER', 'LOC', 'ORG', 'GPE', 'FAC', 'VEH', 'WEA']
    if version.find('kbp') >= 0:
        type_list = ['GPE', 'FAC', 'ORG', 'PER', 'LOC']
    
    count_list = [0] * (len(type_list) + 1)
    for now_df in df:
        n = len(now_df['tokens'])
        count_list[-1] += n * (n + 1) // 2 - len(now_df['entities'])
        for ent in now_df['entities']:
            count_list[type_list.index(ent['type'])] += 1
    return count_list

