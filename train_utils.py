from transformers import AdamW, get_linear_schedule_with_warmup


BERT_ABBV = {'Bio_ClinicalBERT':'clinicalbert',
             'bert-base-multilingual-cased': 'mbert_cased',
             'bert-base-multilingual-uncased': 'mbert_uncased',
             'bluebert_pubmed_mimic_base': 'bluebert',
             'bert-base-cased': 'base_cased',
             'bert-large-cased': 'large_cased',
             'bert-base-uncased': 'base_uncased',
             'bert-large-uncased': 'large_uncased',
             'pubmedbert_abs': 'pubmedbert',
             'scibert_scivocab_uncased': 'scibert',
             'biobert_v1.1': 'biobert',
             'biobert-large-cased-v1.1': 'biobertL',
             'spanbert-large-cased': 'span_large'}


def main_name(bert_name_or_path):
    if bert_name_or_path.lower().find('kebio') >= 0:
        return 'kebio'
    if bert_name_or_path.find('/') == -1:
        if bert_name_or_path in BERT_ABBV:
            return BERT_ABBV[bert_name_or_path]
        return bert_name_or_path
    if bert_name_or_path[-1] == "/":
        bert_name_or_path = bert_name_or_path[:-1]
    name = bert_name_or_path.split('/')[-1]
    if name in BERT_ABBV:
        return BERT_ABBV[name]
    return name

def main_name_list(bert_name_or_path_list):
    return ",".join([main_name(bert_name_or_path) for bert_name_or_path in bert_name_or_path_list.split(',')])

def generate_output_folder_name(args):
    if args.model in ["SpanModel"]:
        args_list = [args.version,
                    args.model,
                    main_name_list(args.bert_name_or_path),
                    args.score]
        if args.negative_sampling:
            args_list += [f"negd_{args.hard_neg_dist}"]
    if args.model in ["SpanAttModel", "SpanAttModelV2", "SpanAttModelV3"]:
        args_list = [args.version,
                    args.model,
                    main_name_list(args.bert_name_or_path),
                    args.class_loss_weight,
                    args.filter_loss_weight,
                    args.span_layer_count,
                    args.max_span_count]
        if args.unscale:
            args_list += ['uns']
    if args.not_correct_bias:
        args_list += ['ncb']
    if args.max_grad_norm != 0.1:
        args_list += [f'norm_{args.max_grad_norm}']
    if args.score in ["tri_attention", "tri_affine"]:
        args_list += [args.att_dim, args.init_std]
        if args.layer_norm:
            args_list += ['ln']
        if args.no_tri_mask:
            args_list += ['ntm']
    # encoder related
    args_list += [args.subword_aggr]
    if args.use_context:
        args_list += ['context']
        if args.context_lstm:
            args_list += ['lstm']
    if args.bert_before_lstm:
        args_list += ['bbl']
    if args.reinit > 0:
        args_list += [f'reinit_{args.reinit}']
    if args.freeze_bert:
        args_list += ['frz']
    if args.rel_pos_attn or args.rel_pos:
        if args.rel_pos_attn:
            args_list += ['relatt']
        if args.rel_pos:
            args_list += ['rel']
        args_list += [args.rel_k]
    if args.word:
        args_list += [f'word_{args.word_dp}']
        if args.word_embed:
            args_list += [f'{args.word_embed}']
        if args.word_freeze:
            args_list += ["wfz"]
    if args.char:
        args_list += [f'char_{args.char_dim}_{args.char_dp}']
    if args.pos:
        args_list += [f'pos_{args.pos_dim}_{args.pos_dp}']
    args_list += [f'{args.agg_layer}_{args.lstm_dim}_{args.lstm_dp}_{args.lstm_layer}']
    if args.bert_output != 'last':
        args_list += [args.bert_output.split('-')[0]]
    if args.act != "relu":
        args_list += [args.act]
    if args.ema > 0.:
        args_list += [f'ema_{args.ema}']
    if args.share_parser:
         args_list += ['sps']
    if args.type_attention:
        args_list += ['type_att']
    if args.token_aux:
        args_list += [f'taux_{args.token_schema}_{args.token_aux_weight}']
    if args.trans_aux:
        args_list += [f'traux_{args.trans_aux_weight}']
    if args.warmup_ratio != 0.1:
        args_list += [f'warm{args.warmup_ratio}']
    if args.aux_loss and args.model.find("DETR") >= 0:
        args_list += ['aux']
    if args.pre_norm:
        args_list += ['prenorm']
    if args.scale != "none":
        args_list += [args.scale]
    if args.weight_scheduler != "none":
        args_list += [args.weight_scheduler]
    if args.loss == "ce":
        if args.na_weight != 1.0:
            args_list += [f"ce_{args.na_weight}"]
    if args.loss != "ce":
        if args.loss == "focal":
            args_list += [f"focal_{args.focal_gamma}_{args.focal_alpha}"]
        elif args.loss == "ldam":
            args_list += [f"ldam_{args.ldam_max_m}_{args.ldam_s}"]
        elif args.loss == "dice":
            args_list += [f"dice_{args.dice_alpha}_{args.dice_gamma}"]
        elif args.loss == "two":
            args_list += [f"two_{args.na_weight}"]
    if float(args.kl_alpha) > 0.0 and args.kl != "none":
        args_list += [f"kl_{args.kl}_{args.kl_alpha}"]
    if float(args.label_smoothing) >= 0.0:
        args_list += [str(args.label_smoothing)]
    args_list += [f'len_{args.truncate_length}',
                  f'epoch_{args.train_epoch}',
                  f'lr_{args.learning_rate}_{args.encoder_learning_rate}_{args.task_learning_rate}',
                  f'bsz_{int(args.batch_size) * int(args.gradient_accumulation_steps)}']
    if args.no_lr_decay:
        args_list += ['nld']
    if args.reduce_last:
        args_list += ['rdl']
    if args.seed != -1:
        args_list += [f's{args.seed}']
    if args.no_linear_class:
        args_list += ['nolin']
    args_list += ['tti'] # token_type_ids
    args_list += [f'mlpdp_{args.dp}']
    args_list += [args.tag]
    output_basename = "-".join([str(arg) for arg in args_list])
    return output_basename
    
    
def generate_optimizer_scheduler(args, model, len_train_dataloader):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = []
    learning_rate = args.learning_rate
    encoder_learning_rate = args.encoder_learning_rate if args.encoder_learning_rate > 0 else learning_rate
    decoder_learning_rate = args.decoder_learning_rate if args.decoder_learning_rate > 0 else learning_rate
    task_learning_rate = args.task_learning_rate if args.task_learning_rate > 0 else learning_rate
    
    bert_params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and n.find('bert') >= 0],
            "weight_decay": args.weight_decay,
            "lr": learning_rate
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and n.find('bert') >= 0],
            "weight_decay": 0.0,
            "lr": learning_rate
        }, 
    ]
    encoder_params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and n.find('encoder') >= 0 and n.find('bert') == -1],
            "weight_decay": args.weight_decay,
            "lr": encoder_learning_rate
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and n.find('encoder') >= 0 and n.find('bert') == -1],
            "weight_decay": 0.0,
            "lr": encoder_learning_rate
        }, 
    ]
    decoder_params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and n.find('decoder') >= 0],
            "weight_decay": args.weight_decay,
            "lr": decoder_learning_rate
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and n.find('decoder') >= 0],
            "weight_decay": 0.0,
            "lr": decoder_learning_rate
        }, 
    ]
    task_params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and n.find('encoder') == -1 and n.find('decoder') == -1],
            "weight_decay": args.weight_decay,
            "lr": task_learning_rate
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and  n.find('encoder') == -1 and n.find('decoder') == -1],
            "weight_decay": 0.0,
            "lr": task_learning_rate
        }, 
    ]
    
    if bert_params[0]['params'] or bert_params[1]['params']:
        optimizer_grouped_parameters.extend(bert_params)
    if encoder_params[0]['params'] or encoder_params[1]['params']:
        optimizer_grouped_parameters.extend(encoder_params)
    if decoder_params[0]['params'] or decoder_params[1]['params']:
        optimizer_grouped_parameters.extend(decoder_params)
    if task_params[0]['params'] or task_params[1]['params']:
        optimizer_grouped_parameters.extend(task_params) 
        
    optimizer = AdamW(optimizer_grouped_parameters, eps=1e-8, correct_bias=not args.not_correct_bias)
    
    if not args.no_lr_decay:
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(args.train_epoch * len_train_dataloader * float(args.warmup_ratio)),
                                                    num_training_steps=args.train_epoch * len_train_dataloader)
    else:
        scheduler = None
    return optimizer, scheduler

def weight_scheduler(epoch_idx, total_epoch=None, args=None, method="square"):
    if method == "square":
        if args is not None:
            total_epoch = args.train_epoch
        return 1 - (epoch_idx / total_epoch) ** 2
    else:
        raise NotImplementedError
    
