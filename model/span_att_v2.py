import torch
from torch import nn
from opt_einsum import contract
import torch.nn.functional as F
from model.parser import Biaffine, TypeAttention, TriAttention, TriAffineParser, TriAffine
from model.text_encoder import TextEncoder
from model.losses import create_loss_function
from span_utils import negative_sampling
from model.transformer import TransformerEncoderLayer_Pos, TransformerEncoder_Pos
from span_utils import tensor_idx_add
from model.mlp import MLP


class SpanAttModelV2(nn.Module):
    def __init__(self, bert_model_path, encoder_config_dict,
                 num_class, score_setting, loss_config):
        super(SpanAttModelV2, self).__init__()
        
        self.encoder = TextEncoder(bert_model_path, encoder_config_dict[0], encoder_config_dict[1], encoder_config_dict[2], encoder_config_dict[3], encoder_config_dict[4])
        self.encoder_config_dict = encoder_config_dict
        self.hidden_dim = self.encoder.bert_hidden_dim

        self.true_class = num_class
        self.num_class = self.true_class + 1
        
        self.score_setting = score_setting
        self.parser_list = nn.ModuleList()
        
        self.dropout = self.score_setting.get('dp', 0.2)
        assert self.score_setting.get('tri_affine', False)
        self.parser_list.append(TriAffineParser(self.hidden_dim,
                                                self.num_class,
                                                self.score_setting.get('att_dim', None),
                                                not self.score_setting['no_tri_mask'],
                                                self.score_setting['reduce_last'],
                                                True,
                                                self.dropout,
                                                self.score_setting['scale'],
                                                self.score_setting['init_std'],
                                                self.score_setting['layer_norm']))
        # self.dropout_list.append(nn.Dropout(self.score_setting.get('dp', 0.2)))
        
        # self.class_loss_fn = nn.CrossEntropyLoss()
        self.loss_config = loss_config
        self.loss_config['true_class'] = self.true_class
        self.class_loss_fn = create_loss_function(self.loss_config)
        
        self._hidden_dim = self.score_setting['att_dim']

        self.max_span_count = self.encoder_config_dict[5].get('max_span_count', 30)
        
        self.act = self.encoder_config_dict[5].get('act', 'relu')
       
        self.linear_h2 = MLP(self._hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout, self.act)
        self.linear_t2 = MLP(self._hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout, self.act)
        self.linear_s2 = MLP(self._hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout, self.act)
        self.linear_h3 = MLP(self._hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout, self.act)
        self.linear_t3 = MLP(self._hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout, self.act)
        self.linear_s3 = MLP(self._hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout, self.act)
        
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
            
        self.filter_loss_weight = self.loss_config.get('filter_loss_weight', 1.0)
        
        self.share_parser = self.encoder_config_dict[5].get('share_parser', False)
        if not self.share_parser:
            self.span_triaffine = TriAffine(self._hidden_dim, self.num_class, init_std=self.score_setting['init_std'])
            self.final_parser = TriAffine(self._hidden_dim, self.num_class, init_std=self.score_setting['init_std'])
            self.final_uni = nn.Parameter(torch.randn(self.num_class))
            
        self.class_loss_weight = None
        
        if self.score_setting.get('layer_norm', False):
            self.norm_h2 = nn.LayerNorm(self._hidden_dim)
            self.norm_t2 = nn.LayerNorm(self._hidden_dim)
            self.norm_s2 = nn.LayerNorm(self._hidden_dim)
            self.norm_h3 = nn.LayerNorm(self._hidden_dim)
            self.norm_t3 = nn.LayerNorm(self._hidden_dim)
            self.norm_s3 = nn.LayerNorm(self._hidden_dim)
            
        if self.loss_config.get('kl', 'none') != 'none':
            self.kl = self.loss_config['kl']
            self.kl_alpha = self.loss_config['kl_alpha']
            self.kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean")

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

        new_span_score, filtered, flat_idx = class_tuple[3], class_tuple[4], class_tuple[5]
        
        bsz = input_ids.size(0)
        max_span_count = flat_idx.size(0) // bsz
        flat_idx = flat_idx.reshape(bsz, max_span_count, 3)

        # sort flat idx
        sort_num = flat_idx[:,:,1] * 200 + flat_idx[:,:,2] * 1
        od = sort_num.sort()[1]
        new_flat_idx = torch.zeros_like(flat_idx)
        for i in range(bsz):
            new_flat_idx[i] = flat_idx[i][od[i]]

        embeds_length = (input_word != max(input_word.view(-1))).sum(1)
        
        if self.loss_config['name'] != "two":
            new_span_class_idx = new_span_score.argmax(-1)
            use_idx = torch.bitwise_and(new_span_class_idx < self.true_class,
                                        new_flat_idx[:,:,-1] < embeds_length.unsqueeze(1)) # b * max_span
        else:
            new_span_class_idx = new_span_score[:,:,0:-1].argmax(-1)
            # we can modify this threshold 0 on dev set for higher f1 score
            use_idx = torch.bitwise_and(new_span_score[:,:,-1] < 0,
                                        new_flat_idx[:,:,-1] < embeds_length.unsqueeze(1)) # b * max_span
        
        result = []     
        for i in range(bsz):
            x = new_flat_idx[i,use_idx[i],1]
            y = new_flat_idx[i,use_idx[i],2]
            cl = new_span_class_idx[i][use_idx[i]]
            result.append([[x[j], y[j], cl[j]] for j in range(x.size(0))])
     
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
        
        # hs_class, alpha, head1, tail1, mid1 = self.parser_list[0](self.dropout_list[0](memory))
        hs_class, alpha, head1, tail1, mid1 = self.parser_list[0](memory)
        # alpha: b * seq * seq * seq * type
        # mid1: batch * seq * _hidden_dim
        # topk_idx b * seq_x, b * seq_y
        # h_span: b * max_span_count * type * _hidden_dim
        
        bsz, seq = hs_class.size(0), hs_class.size(1)
        embeds_length = (input_word != max(input_word.view(-1))).sum(1)
        
        filtered, flat_idx = self.topk(hs_class, self.max_span_count, embeds_length, self.loss_config['name'])

        max_span_count = flat_idx.size(0) // bsz
        topk_alpha = alpha[filtered].reshape(bsz, max_span_count, -1, self.num_class)
        h_span = contract('bmst,bsh->bmth', topk_alpha, mid1) # b * max_span_count * type * _hidden_dim
        bsz, _hidden_dim = h_span.size(0), h_span.size(-1)
        
        head2 = self.linear_h2(head1)
        tail2 = self.linear_t2(tail1)
        h_span2 = self.linear_s2(h_span)
        
        head3 = self.linear_h3(head1)
        tail3 = self.linear_t3(tail1)
        h_span3 = self.linear_s3(h_span)
        
        if self.score_setting.get('layer_norm', False):
            head2 = self.norm_h2(head2)
            tail2 = self.norm_t2(tail2)
            h_span2 = self.norm_s2(h_span2)
            head3 = self.norm_h3(head3)
            tail3 = self.norm_t3(tail3)
            h_span3 = self.norm_s3(h_span3)
        
        x2 = head2.unsqueeze(2).repeat(1,1,seq,1)[filtered].reshape(bsz, max_span_count, -1) # bsz * max_span_count * _hidden
        y2 = tail2.unsqueeze(1).repeat(1,seq,1,1)[filtered].reshape(bsz, max_span_count, -1)
        x2 = torch.cat((x2, torch.ones_like(x2[..., :1])), -1)
        y2 = torch.cat((y2, torch.ones_like(y2[..., :1])), -1)
        x3 = head3.unsqueeze(2).repeat(1,1,seq,1)[filtered].reshape(bsz, max_span_count, -1) # bsz * max_span_count * _hidden
        y3 = tail3.unsqueeze(1).repeat(1,seq,1,1)[filtered].reshape(bsz, max_span_count, -1)
        x3 = torch.cat((x3, torch.ones_like(x3[..., :1])), -1)
        y3 = torch.cat((y3, torch.ones_like(y3[..., :1])), -1)
        
        if not self.share_parser:
            span_score = contract('bxi,bzrk,ikjr,bxj->bzxr', x2, h_span2, self.span_triaffine.weight, y2)
            if self.score_setting['scale'] == "triv2":
                span_score *= self.parser_list[0].parser0.scale_factor
            span_alpha = F.softmax(span_score, dim=-2)
            score = contract('bxi,bzrk,ikjr,bxj->bzxr', x3, h_span3, self.final_parser.weight, y3)
            s = contract('bzxr,bzxr->bzr', score, span_alpha)
            new_span_score = self.final_uni.unsqueeze(0).unsqueeze(0) + s
        else:
            span_score = contract('bxi,bzrk,ikjr,bxj->bzxr', x2, h_span2, self.parser_list[0].parser0.weight, y2)
            if self.score_setting['scale'] == "triv2":
                span_score *= self.parser_list[0].parser0.scale_factor
            span_alpha = F.softmax(span_score, dim=-2)
            score = contract('bxi,bzrk,ikjr,bxj->bzxr', x3, h_span3, self.parser_list[0].parser1.weight, y3)
            s = contract('bzxr,bzxr->bzr', score, span_alpha)
            new_span_score = self.parser_list[0].uni.unsqueeze(0).unsqueeze(0) + s
                
        if self.token_aux_loss:
            token_class = self.linear_token(self.token_dropout(memory))
        else:
            token_class = None
            
        if self.trans_aux_loss:
            trans_class = self.trans_bi(self.trans_dropout(memory))
        else:
            trans_class = None
            
        return (hs_class, token_class, trans_class, new_span_score, filtered, flat_idx)
    
    @torch.no_grad()
    def topk(self, hs_class, count, embeds_length, loss_type='ce'):
        """
        hs_class: Batch * word * word * class
        """
        bsz, word_cnt = hs_class.size(0), hs_class.size(1)
        seq = torch.arange(word_cnt).to(hs_class.device)
        mask_place = torch.bitwise_or((seq.unsqueeze(1) > seq.unsqueeze(0)).unsqueeze(0), (seq.unsqueeze(0) >= embeds_length.unsqueeze(1)).unsqueeze(1))
        count = min(count, word_cnt * word_cnt)
        if loss_type != "two":
            hs_class_sft = hs_class.softmax(-1) # Batch * word * word * class
            hs_class_sft = hs_class_sft.masked_fill(mask_place.unsqueeze(-1) ,value=-1)
            hs_class_prob, _ = hs_class_sft[:,:,:,0:-1].max(dim=-1) # Batch * word * word
            topk_flat_idx = torch.topk(hs_class_prob.reshape(bsz, -1), k=count)[1]
            filtered = torch.zeros_like(hs_class_prob)
        else:
            class_logit = -hs_class[:,:,:,-1]
            class_logit = class_logit.masked_fill(mask_place, value=-1e6) # Batch * word * word
            topk_flat_idx = torch.topk(class_logit.reshape(bsz, -1), k=count)[1]
            filtered = torch.zeros_like(class_logit)
        
        topk_b_idx = torch.arange(bsz).repeat_interleave(count).to(hs_class.device)
        topk_x_idx = (topk_flat_idx // word_cnt).reshape(-1)
        topk_y_idx = (topk_flat_idx % word_cnt).reshape(-1)
        
        flat_idx = torch.stack([topk_b_idx, topk_x_idx, topk_y_idx]).t()
        filtered = tensor_idx_add(filtered, flat_idx).bool()
        return filtered, flat_idx # Batch * count, Batch * count
        
    
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
        if self.class_loss_weight is not None:
            class_loss *= self.class_loss_weight
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
        
        new_span_score, filtered = class_tuple[3], class_tuple[4]
        filtered_loss = self.class_loss_fn(new_span_score.reshape(-1,self.num_class), label[filtered].reshape(-1))
        class_loss += self.filter_loss_weight * filtered_loss
        
        if self.loss_config.get('kl', 'none') != 'none':
            p = new_span_score.reshape(-1,self.num_class)
            q = hs_class[filtered].reshape(-1,self.num_class)
            if self.loss_config['kl'] == 'pq':
                kl_loss = self.kl_loss_fn(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1))
            elif self.loss_config['kl'] == 'qp':
                kl_loss = self.kl_loss_fn(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1))
            elif self.loss_config['kl'] == 'both':
                kl_loss = (self.kl_loss_fn(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1))+self.kl_loss_fn(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1))) / 2
            class_loss += self.kl_alpha * kl_loss

        return class_loss

class SpanAttModelV3(SpanAttModelV2):
    def __init__(self, bert_model_path, encoder_config_dict,
                 num_class, score_setting, loss_config):
        super(SpanAttModelV3, self).__init__(bert_model_path,
                                             encoder_config_dict,
                                             num_class,
                                             score_setting,
                                             loss_config)
        self.linear_h2 = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout, self.act)
        self.linear_t2 = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout, self.act)
        self.linear_s2 = MLP(self._hidden_dim, self._hidden_dim, self._hidden_dim, 1, self.dropout, self.act)
        self.linear_h3 = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout, self.act)
        self.linear_t3 = MLP(self.hidden_dim, self._hidden_dim, self._hidden_dim, 2, self.dropout, self.act)
        self.linear_s3 = MLP(self._hidden_dim, self._hidden_dim, self._hidden_dim, 1, self.dropout, self.act)

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
        
        hs_class, alpha, head1, tail1, mid1 = self.parser_list[0](memory)
        
        bsz, seq = hs_class.size(0), hs_class.size(1)
        embeds_length = (input_word != max(input_word.view(-1))).sum(1)
        
        filtered, flat_idx = self.topk(hs_class, self.max_span_count, embeds_length, self.loss_config['name'])

        max_span_count = flat_idx.size(0) // bsz
        topk_alpha = alpha[filtered].reshape(bsz, max_span_count, -1, self.num_class)
        h_span = contract('bmst,bsh->bmth', topk_alpha, mid1) # b * max_span_count * type * _hidden_dim
        bsz, _hidden_dim = h_span.size(0), h_span.size(-1)
        
        head2 = self.linear_h2(memory)
        tail2 = self.linear_t2(memory)
        h_span2 = self.linear_s2(h_span)
        
        head3 = self.linear_h3(memory)
        tail3 = self.linear_t3(memory)
        h_span3 = self.linear_s3(h_span)
        
        x2 = head2.unsqueeze(2).repeat(1,1,seq,1)[filtered].reshape(bsz, max_span_count, -1) # bsz * max_span_count * _hidden
        y2 = tail2.unsqueeze(1).repeat(1,seq,1,1)[filtered].reshape(bsz, max_span_count, -1)
        x2 = torch.cat((x2, torch.ones_like(x2[..., :1])), -1)
        y2 = torch.cat((y2, torch.ones_like(y2[..., :1])), -1)
        x3 = head3.unsqueeze(2).repeat(1,1,seq,1)[filtered].reshape(bsz, max_span_count, -1) # bsz * max_span_count * _hidden
        y3 = tail3.unsqueeze(1).repeat(1,seq,1,1)[filtered].reshape(bsz, max_span_count, -1)
        x3 = torch.cat((x3, torch.ones_like(x3[..., :1])), -1)
        y3 = torch.cat((y3, torch.ones_like(y3[..., :1])), -1)
        
        if not self.share_parser:
            span_score = contract('bxi,bzrk,ikjr,bxj->bzxr', x2, h_span2, self.span_triaffine.weight, y2)
            if self.score_setting['scale'] == "triv2":
                span_score *= self.parser_list[0].parser0.scale_factor
            span_alpha = F.softmax(span_score, dim=-2)
            score = contract('bxi,bzrk,ikjr,bxj->bzxr', x3, h_span3, self.final_parser.weight, y3)
            s = contract('bzxr,bzxr->bzr', score, span_alpha)
            new_span_score = self.final_uni.unsqueeze(0).unsqueeze(0) + s
        else:
            span_score = contract('bxi,bzrk,ikjr,bxj->bzxr', x2, h_span2, self.parser_list[0].parser0.weight, y2)
            if self.score_setting['scale'] == "triv2":
                span_score *= self.parser_list[0].parser0.scale_factor
            span_alpha = F.softmax(span_score, dim=-2)
            score = contract('bxi,bzrk,ikjr,bxj->bzxr', x3, h_span3, self.parser_list[0].parser1.weight, y3)
            s = contract('bzxr,bzxr->bzr', score, span_alpha)
            new_span_score = self.parser_list[0].uni.unsqueeze(0).unsqueeze(0) + s
                
        if self.token_aux_loss:
            token_class = self.linear_token(self.token_dropout(memory))
        else:
            token_class = None
            
        if self.trans_aux_loss:
            trans_class = self.trans_bi(self.trans_dropout(memory))
        else:
            trans_class = None
            
        return (hs_class, token_class, trans_class, new_span_score, filtered, flat_idx)
