import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from transformers import AutoModel, AutoConfig
import torch
from torch import nn, from_numpy
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model.position_embed import SinusoidalPositionalEmbedding
load_kebiolm = True
try:
    from model.kebiolm.modeling_kebio import KebioModel
    from model.kebiolm.configuration_kebio import KebioConfig
except BaseException:
    load_kebiolm = False

  
def reinit(model, layer_count):
    for layer in model.encoder.layer[-layer_count:]:
        for module in layer.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                print(f'Re init {module}')
                module.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                print(f'Re init {module}')
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
                

class TextEncoder(nn.Module):
    def __init__(self, bert_model_path_list,
                 bert_config={'bert_before_lstm': False,
                              'subword_aggr': 'first'},
                 word_embedding_config={},
                 char_embedding_config={},
                 pos_embedding_config={},
                 lstm_config={}):
        super(TextEncoder, self).__init__()
        self.lstm_input_size = 0
        
        self.bert_list = nn.ModuleList()
        self.bert_config_list = []
        self.bert_model_path_list = bert_model_path_list.split(',')
        for bert_model_path in self.bert_model_path_list:
            print(bert_model_path)
            if bert_model_path.lower().find('kebio') == -1:
                self.bert_list.append(AutoModel.from_pretrained(bert_model_path))
                self.bert_config_list.append(AutoConfig.from_pretrained(bert_model_path))
            else:
                assert load_kebiolm
                config = KebioConfig.from_pretrained(bert_model_path)
                model = KebioModel.from_pretrained(bert_model_path, config=config)
                self.bert_list.append(model)
                self.bert_config_list.append(config)
        self.bert_additional_config = bert_config
        if self.bert_additional_config.get('reinit', 0) > 0:
            for model in self.bert_list:
                reinit(model, self.bert_additional_config.get('reinit', 0))
        
        self.bert_hidden_dim = max([config.hidden_size for config in self.bert_config_list])
        self.all_bert_hidden_dim = sum([config.hidden_size for config in self.bert_config_list])
        
        if self.bert_additional_config['bert_output'] == 'concat-last-4':
            self.bert_hidden_dim *= 4
            self.all_bert_hidden_dim *= 4
            
        if self.bert_additional_config['bert_before_lstm']:
            self.lstm_input_size += self.all_bert_hidden_dim
            self.reduce_dim = nn.Linear(lstm_config['dim'], self.bert_hidden_dim)
        else:
            self.reduce_dim = nn.Linear(lstm_config['dim'] + self.all_bert_hidden_dim, self.bert_hidden_dim)

        self.word_embedding_config = word_embedding_config
        if self.word_embedding_config:
            embedding_weight = np.load(word_embedding_config['path'])
            padding_idx = word_embedding_config['padding_idx'] if word_embedding_config['padding_idx'] < embedding_weight.shape[0] else None
            self.word_embedding = nn.Embedding(embedding_weight.shape[0],
                                               embedding_weight.shape[1],
                                               padding_idx=padding_idx)
            self.word_embedding.weight.data.copy_(from_numpy(embedding_weight))
            self.word_dropout = nn.Dropout(word_embedding_config['dropout'])
            self.lstm_input_size += word_embedding_config['dim']
            if word_embedding_config['freeze']:
                self.word_embedding.weight.requires_grad = False

        self.char_embedding_config = char_embedding_config
        if self.char_embedding_config:
            self.char_embedding = nn.Embedding(200,
                                               char_embedding_config['dim'],
                                               padding_idx=char_embedding_config['padding_idx'])
            self.char_dropout = nn.Dropout(char_embedding_config['dropout'])
            if char_embedding_config['layer'] == 1:
                char_embedding_config['dropout'] = 0.0
            self.char_lstm = nn.LSTM(input_size=char_embedding_config['dim'],
                                     hidden_size=char_embedding_config['dim'] // 2,
                                     num_layers=char_embedding_config['layer'],
                                     bidirectional=True,
                                     dropout=char_embedding_config['dropout'],
                                     batch_first=True)
            self.lstm_input_size += char_embedding_config['dim']

        self.pos_embedding_config = pos_embedding_config
        if self.pos_embedding_config:
            self.pos_embedding = nn.Embedding(1100,
                                              pos_embedding_config['dim'],
                                              padding_idx=pos_embedding_config['padding_idx'])
            self.pos_dropout = nn.Dropout(pos_embedding_config['dropout'])
            self.lstm_input_size += pos_embedding_config['dim']

        self.lstm_config = lstm_config
        if self.lstm_config:
            if self.lstm_config['name'] == "lstm":
                self.lstm = nn.LSTM(input_size=self.lstm_input_size,
                                    hidden_size=lstm_config['dim'] // 2,
                                    num_layers=lstm_config['layer'],
                                    bidirectional=True,
                                    dropout=lstm_config['dropout'],
                                    batch_first=True)
            elif self.lstm_config['name'] == "transformer":
                self.position_embedding = SinusoidalPositionalEmbedding(200, self.lstm_input_size)
                if lstm_config['dim'] == 512:
                    nhead = 8
                if lstm_config['dim'] == 768:
                    nhead = 12
                if lstm_config['dim'] == 1024:
                    nhead = 16
                self.input_reduce = nn.Linear(self.lstm_input_size, lstm_config['dim'])
                encoder_layer = nn.TransformerEncoderLayer(d_model=lstm_config['dim'],
                                                           nhead=nhead) # pytorch 1.7.0 do not support batch_first
                layernorm = nn.LayerNorm(lstm_config['dim'])
                self.trans = nn.TransformerEncoder(encoder_layer, num_layers=lstm_config['layer'], norm=layernorm)
            self.context_lstm = self.lstm_config['context_lstm']
            
            
    def lstm_forward(self, x, lengths, lstm):
        np_lengths = lengths.cpu().numpy()
        np_lengths[np_lengths==0] = 1
        x_pack = pack_padded_sequence(x, np_lengths, batch_first=True, enforce_sorted=False)
        h_pack, _ = lstm(x_pack)
        h, _ = pad_packed_sequence(h_pack, batch_first=True)
        return h
    
    
    def combine(self, hidden, subword_group, agg):
        '''
        hidden: bsz * seq * hidden
        subword_group: bsz * seq1 * seq
        '''
        assert agg in ['max', 'mean']
        
        if agg == "mean":
            size = subword_group.sum(-1).unsqueeze(-1) + 1e-20 # bsz * seq1 * 1
            sup = subword_group.unsqueeze(-1) * hidden.unsqueeze(1) # bsz * seq1 * seq * hidden
            sup = sup.sum(dim=2) / size # bsz * seq1 * hidden
        elif agg == "max":
            m = ((1 - subword_group.float()) * (-1e20)).unsqueeze(-1) # bsz * seq1 * seq * 1
            sup = m + hidden.unsqueeze(1) # bsz * seq1 * seq * hidden
            sup = sup.max(dim=2)[0]
            sup[sup==-1e20] = 0
            
        return sup

    def get_bert_hidden(self, input_ids, attention_mask=None, ce_mask=None, token_type_ids=None, subword_group=None, 
                        context_ce_mask=None, context_subword_group=None, bert_embed=None):
        if bert_embed is not None:
            return bert_embed
        
        all_bert_hidden = []
        if len(input_ids.size()) == 2:
            input_ids = input_ids.unsqueeze(1)
            attention_mask = attention_mask.unsqueeze(1)
            token_type_ids = token_type_ids.unsqueeze(1)
            ce_mask = ce_mask.unsqueeze(1)
            context_ce_mask = context_ce_mask.unsqueeze(1)

        # for old version code
        if not hasattr(self, 'bert_list'):
            self.bert_list = nn.ModuleList()
            self.bert_list.append(self.bert)
            self.bert_config_list = [self.bert_config]

        for idx, bert in enumerate(self.bert_list):
            if load_kebiolm and isinstance(bert, KebioModel):
                if self.bert_additional_config['bert_output'] == 'last':
                    memory = bert(input_ids[:,idx], attention_mask[:,idx], token_type_ids=token_type_ids[:,idx])[2]  # Batch * Length * Hidden
                else:
                    raise NotImplementedError
            else:
                if self.bert_additional_config['bert_output'] == 'last':
                    memory = bert(input_ids[:,idx], attention_mask[:,idx], token_type_ids=token_type_ids[:,idx])[0]  # Batch * Length * Hidden
                else: 
                    opt = bert(input_ids[:,idx], attention_mask[:,idx], token_type_ids=token_type_ids[:,idx], return_dict=True, output_hidden_states=True)['hidden_states']
                    if self.bert_additional_config['bert_output'] == 'mean-last-4':
                        memory = torch.stack(opt[-4:],dim=-1).mean(-1)
                    elif self.bert_additional_config['bert_output'] == 'concat-last-4':
                        memory = torch.cat(opt[-4:],dim=-1)
        
            if not self.context_lstm:
                if self.bert_additional_config['subword_aggr'] == 'first':
                    # index.shape = Batch * 160 * Hidden
                    ce_length = (ce_mask[:,idx]==1).sum(-1)
                    arange = torch.arange(input_ids.size(-1))
                    index = torch.zeros((input_ids.size(0), 200), dtype=torch.int64).to(input_ids.device)
                    for i in range(input_ids.size(0)):
                        index[i][0:ce_length[i]] = arange[ce_mask[:,idx][i]==1]
                    index = index.unsqueeze(-1).repeat(1,1,self.bert_config_list[idx].hidden_size)
                    bert_hidden = torch.gather(memory, dim=1, index=index)
                elif self.bert_additional_config['subword_aggr'] == 'max':
                    if len(subword_group.size()) == 3:
                        sbg = subword_group
                    else:
                        sbg = subword_group[:,idx]
                    bert_hidden = self.combine(memory, sbg, 'max')
                elif self.bert_additional_config['subword_aggr'] == 'mean':
                    if len(subword_group.size()) == 3:
                        sbg = subword_group
                    else:
                        sbg = subword_group[:,idx]
                    bert_hidden = self.combine(memory, sbg, 'mean')
            else:
                if self.bert_additional_config['subword_aggr'] == 'first':
                    ce_length = (context_ce_mask[:,idx]==1).sum(-1)
                    arange = torch.arange(input_ids.size(-1))
                    index = torch.zeros((input_ids.size(0), 200 * 3), dtype=torch.int64).to(input_ids.device)
                    for i in range(input_ids.size(0)):
                        index[i][0:ce_length[i]] = arange[context_ce_mask[:,idx][i]==1]
                    index = index.unsqueeze(-1).repeat(1,1,self.bert_config_list[idx].hidden_size)
                    bert_hidden = torch.gather(memory, dim=1, index=index)
                elif self.bert_additional_config['subword_aggr'] == 'max':
                    if len(subword_group.size()) == 3:
                        sbg = context_subword_group
                    else:
                        sbg = context_subword_group[:,idx]
                    bert_hidden = self.combine(memory, sbg, 'max')
                elif self.bert_additional_config['subword_aggr'] == 'mean':
                    if len(subword_group.size()) == 3:
                        sbg = context_subword_group
                    else:
                        sbg = context_subword_group[:,idx]
                    bert_hidden = self.combine(memory, sbg, 'mean')
                
            all_bert_hidden.append(bert_hidden)
        
        return torch.cat(all_bert_hidden, dim=-1)
    
    def get_pos_word_char(self, pos, word, char):
        embeds = []
        if self.pos_embedding_config:
            pos_embed = self.pos_embedding(pos)
            pos_embed = self.pos_dropout(pos_embed)
            embeds.append(pos_embed)
        if self.word_embedding_config:
            word_embed = self.word_embedding(word)
            word_embed = self.word_dropout(word_embed)
            embeds.append(word_embed)
        if self.char_embedding_config:
            # input_char: Batch * max_word_count * max_char_count
            bsz, word_cnt, ch_cnt = char.size()
            input_char = char.view(-1, ch_cnt)
            input_char_embed = self.char_embedding(input_char)
            input_char_embed = self.char_dropout(input_char_embed)
            input_char_mask = input_char != self.char_embedding_config['padding_idx']
            char_length = input_char_mask.sum(-1)
            char_length[char_length==0] = 1
            # (Batch * max_word_count, max_char_count, char_dim)
            char_embed = self.lstm_forward(input_char_embed, char_length, self.char_lstm)
            char_idx = (char_length - 1).unsqueeze(-1).unsqueeze(-1).repeat(1,1,self.char_embedding_config['dim'])
            last_char_embed = torch.gather(char_embed, 1, char_idx).squeeze(1).reshape(bsz, word_cnt, -1)
            embeds.append(last_char_embed)
        return embeds
            
    def get_embedding(self, input_ids, attention_mask, ce_mask, token_type_ids, subword_group,
                      context_ce_mask, context_subword_group, context_map,
                      input_word, input_char, input_pos,
                      l_input_word, l_input_char, l_input_pos,
                      r_input_word, r_input_char, r_input_pos,
                      bert_embed):
        bert_hidden = self.get_bert_hidden(input_ids, attention_mask, ce_mask, token_type_ids, subword_group, 
                                           context_ce_mask, context_subword_group, bert_embed)
        
        embeds = []
        if self.bert_additional_config['bert_before_lstm']:
            embeds = [bert_hidden]
        
        embeds.extend(self.get_pos_word_char(input_pos, input_word, input_char))
        
        if not self.context_lstm:
            return bert_hidden, embeds

        l_embeds = self.get_pos_word_char(l_input_pos, l_input_word, l_input_char)
        r_embeds = self.get_pos_word_char(r_input_pos, r_input_word, r_input_char)

        return bert_hidden, (embeds, l_embeds, r_embeds)

    def lstm_encode(self, bert_hidden, embeds, embeds_length, input_word):
        if not embeds:
            return bert_hidden[:,0:max(embeds_length),:]
        concat_embeds = torch.cat(embeds, dim=-1)
        if self.lstm_config.get('name', 'lstm') == 'lstm':
            lstm_embeds = self.lstm_forward(concat_embeds, embeds_length, self.lstm)
        elif self.lstm_config['name'] == "transformer":
            concat_embeds = concat_embeds[:,0:max(embeds_length)]
            concat_embeds = self.input_reduce(concat_embeds + self.position_embedding(concat_embeds.size()))
            src_mask = (input_word == max(input_word.view(-1)))[:,0:max(embeds_length)] # True is masked
            lstm_embeds = self.trans(concat_embeds.permute(1,0,2), src_key_padding_mask=src_mask).permute(1,0,2)
        
        if self.bert_additional_config['bert_before_lstm']:
            lstm_embeds = self.reduce_dim(lstm_embeds) # Batch * max_word_count * hidden
            return lstm_embeds
        lstm_embeds = torch.cat([bert_hidden[:,0:lstm_embeds.size(1)], lstm_embeds], dim=-1)
        lstm_embeds = self.reduce_dim(lstm_embeds) # Batch * max_word_count * hidden
        return lstm_embeds
    
    def context_lstm_encode(self, bert_hidden, embeds, embeds_length, input_word, 
                            l_input_word, r_input_word,
                            ce_mask, context_ce_mask, context_map):
        embeds, l_embeds, r_embeds = embeds
        index = context_map.unsqueeze(-1).repeat(1,1,self.all_bert_hidden_dim)
        #import ipdb; ipdb.set_trace()
      
        #problem here
        sen_bert_hidden = torch.gather(bert_hidden, dim=1, index=index)
        bsz = bert_hidden.size(0)
        if not embeds:
            return sen_bert_hidden
        
        # concat embeds l_embeds r_embeds to all_concat_embeds, all_embeds_length
        #print(l_input_word.shape, r_input_word.shape, max(input_word.view(-1)))
        
        #import ipdb; ipdb.set_trace()
        #l_embeds_length = (l_input_word != max(input_word.view(-1))).sum(1)
        #r_embeds_length = (r_input_word != max(input_word.view(-1))).sum(1)
        
        if self.bert_additional_config['bert_before_lstm']:
            embeds = embeds[1:]
        embeds = torch.cat(embeds, dim=-1)
        l_embeds = torch.cat(l_embeds, dim=-1)
        r_embeds = torch.cat(r_embeds, dim=-1)
        
        word_dim = self.lstm_input_size - self.bert_hidden_dim
        # all_concat_embeds = torch.zeros((bsz, sen_bert_hidden.size(1), word_dim)).to(bert_hidden.device)
        all_embeds_length = torch.zeros_like(embeds_length)
        all_concat_list = []
        for i in range(bsz):
            ce_start_idx = torch.nonzero(ce_mask[i])[0][0]
            l = min(l_embeds.size(1), min(context_ce_mask[i][0:ce_start_idx].sum(), sen_bert_hidden.size(1)))
            m = min(embeds.size(1), max(0, min(ce_mask[i].sum(), sen_bert_hidden.size(1)-l)))
            r = min(r_embeds.size(1), max(0, min(context_ce_mask[i].sum()-l-m, sen_bert_hidden.size(1)-l-m)))
            
            if m != ce_mask[i].sum() or m != embeds_length[i]:
                print(m, ce_mask[i].sum(), embeds_length[i])
                import ipdb; ipdb.set_trace()
            
            #print(l,m,r,l+m+r,i,word_dim,embeds_length) 
            all_embeds_length[i] = l + m + r

            # if l:
            #     all_concat_embeds[i,0:l] = l_embeds[i,-l:]
            # all_concat_embeds[i,l:l+m] = embeds[i,0:m]
            # all_concat_embeds[i,l+m:l+m+r] = r_embeds[i,0:r]
            if l:
                all_concat_list.append(F.pad(torch.cat([l_embeds[i,-l:], embeds[i,0:m], r_embeds[i,0:r]],dim=0), (0,0,0,sen_bert_hidden.size(1)-all_embeds_length[i])))
            else:
                all_concat_list.append(F.pad(torch.cat([embeds[i,0:m], r_embeds[i,0:r]],dim=0),(0,0,0,sen_bert_hidden.size(1)-all_embeds_length[i])))
                
            #import ipdb; ipdb.set_trace()

        all_concat_embeds = torch.stack(all_concat_list, dim=0)
        all_concat_embeds = torch.cat([all_concat_embeds, sen_bert_hidden], dim=-1)
        
        if self.lstm_config['name'] == "transformer":
            raise NotImplementedError
        elif self.lstm_config.get('name', 'lstm') == 'lstm':
            lstm_embeds = self.lstm_forward(all_concat_embeds, all_embeds_length, self.lstm)
            
        #import ipdb; ipdb.set_trace()
            
        if self.bert_additional_config['bert_before_lstm']:
            lstm_embeds = self.reduce_dim(lstm_embeds) # Batch * max_word_count * hidden
            #import ipdb; ipdb.set_trace()
            return torch.gather(lstm_embeds, dim=1, index=index)[:,0:max(embeds_length)]
        
        lstm_index = context_map.unsqueeze(-1).repeat(1,1,lstm_embeds.size(-1))
        sen_lstm_embeds = torch.gather(lstm_embeds, dim=1, index=lstm_index)[:,0:max(embeds_length)]
        
        lstm_embeds = torch.cat([sen_bert_hidden[:,0:sen_lstm_embeds.size(1)], sen_lstm_embeds], dim=-1)
        lstm_embeds = self.reduce_dim(lstm_embeds) # Batch * max_word_count * hidden
        return lstm_embeds
        
    def forward(self, input_ids, attention_mask=None, ce_mask=None, token_type_ids=None, subword_group=None,
                context_ce_mask=None, context_subword_group=None, context_map=None,
                input_word=None, input_char=None, input_pos=None, 
                l_input_word=None, l_input_char=None, l_input_pos=None, 
                r_input_word=None, r_input_char=None, r_input_pos=None, 
                bert_embed=None):
        #embeds_length = (input_word != self.word_embedding_config['padding_idx']).sum(1)
        embeds_length = (input_word != max(input_word.view(-1))).sum(1)
        bert_hidden, embeds = self.get_embedding(input_ids, attention_mask, ce_mask, token_type_ids, subword_group,
                                                 context_ce_mask, context_subword_group, context_map,
                                                 input_word, input_char, input_pos,
                                                 l_input_word, l_input_char, l_input_pos,
                                                 r_input_word, r_input_char, r_input_pos,
                                                 bert_embed)
        if not self.context_lstm:
            hidden = self.lstm_encode(bert_hidden, embeds, embeds_length, input_word)
        else:
            hidden = self.context_lstm_encode(bert_hidden, embeds, embeds_length, input_word, 
                                              l_input_word, r_input_word, 
                                              ce_mask, context_ce_mask, context_map)
        #import ipdb; ipdb.set_trace()
        return hidden
