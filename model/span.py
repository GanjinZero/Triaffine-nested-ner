import torch
from torch import nn
from opt_einsum import contract
import torch.nn.functional as F
from model.parser import Biaffine, TypeAttention, TriAttention, TriAffineParser
from model.parser import TriAffineParserWithoutLable, TriAffineParserWithoutBoundary, TriAffineParserWithoutScorer, TriAffineParserWithoutScorerPlusBoundary, LinearTriParser, TriAffineParserLinAttn
from model.text_encoder import TextEncoder
from model.losses import create_loss_function
from span_utils import negative_sampling


class SpanModel(nn.Module):
    def __init__(self, bert_model_path, encoder_config_dict,
                 num_class, score_setting, loss_config):
        super(SpanModel, self).__init__()
        
        self.encoder = TextEncoder(bert_model_path, encoder_config_dict[0], encoder_config_dict[1], encoder_config_dict[2], encoder_config_dict[3], encoder_config_dict[4])
        self.hidden_dim = self.encoder.bert_hidden_dim

        self.true_class = num_class
        self.num_class = self.true_class + 1
        
        self.score_setting = score_setting
        self.parser_list = nn.ModuleList()
        if self.score_setting.get('biaffine', False):
            self.parser_list.append(Biaffine(self.hidden_dim, self.num_class, self.score_setting.get('dp', 0.2)))
        # if self.score_setting.get('type_attention', False):
        #     self.parser_list.append(TypeAttention(self.hidden_dim, self.num_class))
        # if self.score_setting.get('tri_attention', False):
        #     self.parser_list.append(TriAttention(self.hidden_dim,
        #                                          self.num_class,
        #                                          self.score_setting.get('att_dim', None),
        #                                          not self.score_setting['no_tri_mask'],
        #                                          self.score_setting['reduce_last'],
        #                                          self.score_setting.get('dp', 0.2)))
        if self.score_setting.get('tri_affine', False):
            # if not self.score_setting.get('pos_tri', False):
            #     self.parser_list.append(TriAffineParser(self.hidden_dim,
            #                                         self.num_class,
            #                                         self.score_setting.get('att_dim', None),
            #                                         not self.score_setting['no_tri_mask'],
            #                                         self.score_setting['reduce_last'],
            #                                         False,
            #                                         self.score_setting.get('dp', 0.2),
            #                                         self.score_setting['scale'],
            #                                         self.score_setting['rel_pos_attn'],
            #                                         self.score_setting['rel_pos'],
            #                                         self.score_setting['rel_k']))
            self.parser_list.append(TriAffineParser(self.hidden_dim,
                                                    self.num_class,
                                                    self.score_setting.get('att_dim', None),
                                                    not self.score_setting['no_tri_mask'],
                                                    self.score_setting['reduce_last'],
                                                    False,
                                                    self.score_setting.get('dp', 0.2),
                                                    self.score_setting['scale'],
                                                    self.score_setting['init_std'],
                                                    self.score_setting['layer_norm']))
        if self.score_setting.get('tri_affine_wo_label', False):
            self.parser_list.append(TriAffineParserWithoutLable(self.hidden_dim,
                                                 self.num_class,
                                                 self.score_setting.get('att_dim', None),
                                                 not self.score_setting['no_tri_mask'],
                                                 self.score_setting['reduce_last'],
                                                 False,
                                                 self.score_setting.get('dp', 0.2),
                                                 self.score_setting['scale']))
        if self.score_setting.get('tri_affine_wo_boundary', False):
            self.parser_list.append(TriAffineParserWithoutBoundary(self.hidden_dim,
                                                 self.num_class,
                                                 self.score_setting.get('att_dim', None),
                                                 not self.score_setting['no_tri_mask'],
                                                 self.score_setting['reduce_last'],
                                                 False,
                                                 self.score_setting.get('dp', 0.2),
                                                 self.score_setting['scale']))
        if self.score_setting.get('tri_affine_wo_scorer', False):
            self.parser_list.append(TriAffineParserWithoutScorer(self.hidden_dim,
                                                 self.num_class,
                                                 self.score_setting.get('att_dim', None),
                                                 not self.score_setting['no_tri_mask'],
                                                 self.score_setting['reduce_last'],
                                                 False,
                                                 self.score_setting.get('dp', 0.2),
                                                 self.score_setting['scale']))
        if self.score_setting.get('tri_affine_wo_scorer_w_boundary', False):
            self.parser_list.append(TriAffineParserWithoutScorerPlusBoundary(self.hidden_dim,
                                                 self.num_class,
                                                 self.score_setting.get('att_dim', None),
                                                 not self.score_setting['no_tri_mask'],
                                                 self.score_setting['reduce_last'],
                                                 False,
                                                 self.score_setting.get('dp', 0.2),
                                                 self.score_setting['scale']))
        if self.score_setting.get('lineartri', False):
            self.parser_list.append(LinearTriParser(self.hidden_dim,
                                                 self.num_class,
                                                 self.score_setting.get('att_dim', None),
                                                 not self.score_setting['no_tri_mask'],
                                                 self.score_setting['reduce_last'],
                                                 False,
                                                 self.score_setting.get('dp', 0.2)))
        if self.score_setting.get('linattntri', False):
            self.parser_list.append(TriAffineParserLinAttn(self.hidden_dim,
                                                 self.num_class,
                                                 self.score_setting.get('att_dim', None),
                                                 not self.score_setting['no_tri_mask'],
                                                 self.score_setting['reduce_last'],
                                                 False,
                                                 self.score_setting.get('dp', 0.2)))
                        
        self.loss_config = loss_config
        self.loss_config['true_class'] = self.true_class
        self.class_loss_fn = create_loss_function(self.loss_config)
       
        self.token_aux_loss = False
        if 'token_schema' in self.loss_config:
            self.token_aux_loss = True
            self.token_schema = loss_config['token_schema']
            if self.token_schema == "BE":
                self.token_label_count = 2
            elif self.token_schema == "BIE":
                self.token_label_count = 3
            elif self.token_schema == "BIES":
                self.token_label_count = 4
            elif self.token_schema == "BE-type":
                self.token_label_count = 2 * self.true_class
            elif self.token_schema == "BIE-type":
                self.token_label_count = 3 * self.true_class
            elif self.token_schema == "BIES-type":
                self.token_label_count = 4 * self.true_class
            self.linear_token = nn.Linear(self.hidden_dim, self.token_label_count)
            self.token_aux_weight = loss_config['token_aux_weight']
            self.token_dropout = nn.Dropout(self.score_setting.get('dp', 0.2))
            
        self.negative_sampling = False
        if self.loss_config.get('negative_sampling', False):
            self.negative_sampling = True
            self.hard_neg_dist = self.loss_config['hard_neg_dist']
            
        self.trans_aux_loss = False
        if self.loss_config.get('trans_aux', False):
            self.trans_aux_loss = True
            self.trans_bi = Biaffine(self.hidden_dim, 2)
            self.trans_dropout = nn.Dropout(self.score_setting.get('dp', 0.2))
            self.trans_aux_weight = loss_config['trans_aux_weight']
        
    def predict(self, input_ids, attention_mask, ce_mask, token_type_ids, subword_group,
                           context_ce_mask, context_subword_group, context_map,
                           input_word, input_char, input_pos,
                           l_input_word, l_input_char, l_input_pos, 
                           r_input_word, r_input_char, r_input_pos, 
                           bert_embed=None):
        class_tuple = self.get_class_position(input_ids, attention_mask, ce_mask, token_type_ids, subword_group,
                              context_ce_mask, context_subword_group, context_map,
                              input_word, input_char, input_pos, 
                              l_input_word, l_input_char, l_input_pos, 
                              r_input_word, r_input_char, r_input_pos, 
                              bert_embed)        
        
        hs_class = class_tuple[0]
        if self.loss_config['name'] != "two":       
            hs_class_prob, hs_class_idx = torch.max(hs_class, dim=-1) # B * S * Q
        else:
            hs_class_prob, hs_class_idx = torch.max(hs_class[:,:,:,:-1], dim=-1)
            
        result = []
        bsz, seq_length = hs_class_idx.size(0), hs_class_idx.size(1)
        seq = torch.arange(seq_length).to(input_ids.device)
       
        if self.loss_config['name'] != "two":       
            use_idx = torch.bitwise_and((seq.unsqueeze(0) >= seq.unsqueeze(1)).unsqueeze(0), hs_class_idx < self.true_class)
        else:
            use_idx = torch.bitwise_and((seq.unsqueeze(0) >= seq.unsqueeze(1)).unsqueeze(0), hs_class[:,:,:,-1] < 0)
        seq_x = seq.unsqueeze(1).repeat(1, seq_length)
        seq_y = seq.unsqueeze(0).repeat(seq_length, 1)

        for i in range(bsz):
            x = seq_x[use_idx[i]]
            y = seq_y[use_idx[i]]
            cl = hs_class_idx[i][use_idx[i]]
            prob = hs_class_prob[i][use_idx[i]]
            result.append([[x[j], y[j], cl[j], prob[j]] for j in range(x.size(0))])

        return result
    
    def get_class_position(self, input_ids, attention_mask, ce_mask, token_type_ids, subword_group,
                           context_ce_mask, context_subword_group, context_map,
                           input_word, input_char, input_pos,
                           l_input_word, l_input_char, l_input_pos, 
                           r_input_word, r_input_char, r_input_pos, 
                           bert_embed=None):
        memory = self.encoder(input_ids, attention_mask, ce_mask, token_type_ids, subword_group,
                              context_ce_mask, context_subword_group, context_map,
                              input_word, input_char, input_pos, 
                              l_input_word, l_input_char, l_input_pos, 
                              r_input_word, r_input_char, r_input_pos, 
                              bert_embed)
        for idx, parser in enumerate(self.parser_list):
            if idx == 0:
                hs_class = parser(memory)
            else:
                hs_class += parser(memory)
                
        if self.token_aux_loss:
            token_class = self.linear_token(self.token_dropout(memory))
        else:
            token_class = None
            
        if self.trans_aux_loss:
            trans_class = self.trans_bi(self.trans_dropout(memory))
        else:
            trans_class = None
            
        return (hs_class, token_class, trans_class)
    
    def forward(self, input_ids, attention_mask, ce_mask, token_type_ids, subword_group, input_word, input_char, input_pos, 
                label, context_ce_mask=None, context_subword_group=None, context_map=None,
                l_input_word=None, l_input_char=None, l_input_pos=None, 
                r_input_word=None, r_input_char=None, r_input_pos=None,  
                token_label=None, bert_embed=None, head_trans=None, tail_trans=None):
        class_tuple = self.get_class_position(input_ids, attention_mask, ce_mask, token_type_ids, subword_group,
                              context_ce_mask, context_subword_group, context_map,
                              input_word, input_char, input_pos, 
                              l_input_word, l_input_char, l_input_pos, 
                              r_input_word, r_input_char, r_input_pos, 
                              bert_embed)

        # deal with mask
        # hs_class: Batch * word * word * class
        # label: Batch * max_word * max_word
        hs_class = class_tuple[0]
        word_cnt = hs_class.size(1)
        label = label[:,0:word_cnt,0:word_cnt]
        if self.negative_sampling:
            label = negative_sampling(label, self.hard_neg_dist)
            
        class_loss = self.class_loss_fn(hs_class.reshape(-1,self.num_class), label.reshape(-1))
        if self.token_aux_loss:
            token_class = class_tuple[1]
            token_class = token_class.reshape(-1, self.token_label_count)
            token_label = token_label[:,0:word_cnt].reshape(-1, self.token_label_count)
            token_mask = token_label[:,0] >= 0
            token_loss = nn.BCEWithLogitsLoss()(token_class[token_mask], token_label[token_mask].float())
            class_loss += self.token_aux_weight * token_loss
            
        if self.trans_aux_loss:
            trans_class = class_tuple[2][:,0:word_cnt,0:word_cnt] # b * s * s * 2
            head_trans = head_trans[:,0:word_cnt,0:word_cnt].float()
            tail_trans = tail_trans[:,0:word_cnt,0:word_cnt].float()
            trans_mask = head_trans >= 0  # b * s * s
            head_trans_loss = nn.BCEWithLogitsLoss()(trans_class[trans_mask][:,0], head_trans[trans_mask])
            tail_trans_loss = nn.BCEWithLogitsLoss()(trans_class[trans_mask][:,1], tail_trans[trans_mask])
            class_loss += self.trans_aux_weight * (head_trans_loss + tail_trans_loss)
        return class_loss
