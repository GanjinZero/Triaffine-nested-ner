import torch


def prepare_input(batch, args=None, train=True, accelerator=False, device=None):
    if args is not None:
        device = args.device
    batch_lst = []
    if not accelerator:
        for i in range(len(batch)):
            batch_lst.append(batch[i].to(device))
    else:
        batch_lst = batch
        
    inputs = {'input_ids': batch_lst[0],
              'attention_mask': batch_lst[1],
              'ce_mask':batch_lst[2],
              'token_type_ids':batch_lst[3],
              'subword_group':batch_lst[4],
              'context_ce_mask':batch_lst[5],
              'context_subword_group':batch_lst[6],
              'context_map':batch_lst[7],
              'input_word':batch_lst[8],
              'input_char':batch_lst[9],
              'input_pos':batch_lst[10],
              'l_input_word':batch_lst[11],
              'l_input_char':batch_lst[12],
              'l_input_pos':batch_lst[13],              
              'r_input_word':batch_lst[14],
              'r_input_char':batch_lst[15],
              'r_input_pos':batch_lst[16]}
    if train:
        inputs['label'] = batch_lst[17]
        if args.model == "TokenModel" or args.token_aux:
            inputs['token_label'] = batch_lst[18]
        if args.trans_aux:
            inputs['head_trans'] = batch_lst[22]
            inputs['tail_trans'] = batch_lst[23]
    if args.freeze_bert:
        inputs['bert_embed'] = batch_lst[-1]
    return inputs
