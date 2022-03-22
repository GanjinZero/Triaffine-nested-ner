import argparse


def generate_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--version", type=str, default="genia",
                        choices=["ace05", "ace04", "genia91", "kbp"],
                        help="Dataset version.")
    parser.add_argument("--model", type=str, default="SpanModel",
                        choices=["SpanModel", "SpanAttModelV3"])
    parser.add_argument("--schema", type=str, default="span",
                        choices=["span"])
    parser.add_argument("--soft_iou", type=float, default=0.7)
    
    parser.add_argument("--token_schema", type=str, default="BE",
                        choices=['BE', 'BIE', 'BIES', 'BE-type', 'BIE-type', 'BIES-type'])
    parser.add_argument("--token_aux", action="store_true")
    parser.add_argument("--token_aux_weight", type=float, default=1.0)
    
    parser.add_argument("--trans_aux", action="store_true")
    parser.add_argument("--trans_aux_weight", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=-1)
    
    parser.add_argument("--score", type=str, default="biaffine",
                        choices=['biaffine', 'tri_attention', 'tri_affine',
                                 'tri_affine_wo_label', 'tri_affine_wo_boundary',
                                 'tri_affine_wo_scorer', 'tri_affine_wo_scorer_w_boundary',
                                 'lineartri', 'linattntri'])
    
    parser.add_argument("--rel_pos_attn", action="store_true")
    parser.add_argument("--rel_pos", action="store_true")
    parser.add_argument("--rel_k", type=int, default=64)
    
    parser.add_argument("--att_dim", type=int, default=0)
    parser.add_argument("--no_tri_mask", action="store_true")
    parser.add_argument("--reduce_last", action="store_true")
    
    parser.add_argument("--type_attention", action="store_true")    
    parser.add_argument("--aux_loss", action="store_true")  
    
    parser.add_argument("--bert_name_or_path", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    
    parser.add_argument("--freeze_bert", action="store_true")
    parser.add_argument("--no_lr_decay", action="store_true")

    # Train setting
    parser.add_argument("--save_every_epoch", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--encoder_learning_rate", type=float, default=0.0)
    parser.add_argument("--decoder_learning_rate", type=float, default=0.0)
    parser.add_argument("--task_learning_rate", type=float, default=0.0)
    parser.add_argument("--not_correct_bias", action="store_true")
    
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--train_epoch", type=int, default=100)
    parser.add_argument("--early_stop_epoch", type=int, default=-1)
    parser.add_argument("--truncate_length", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=0.1)
    parser.add_argument("--output_base_dir", type=str, default="./output/")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--ema", type=float, default=0.)
    
    # DETR
    parser.add_argument("--query_head_count", type=int, default=30)
    parser.add_argument("--decoder_layer_count", type=int, default=4)
    parser.add_argument("--class_loss_weight", type=float, default=1.0)
    parser.add_argument("--filter_loss_weight", type=float, default=1.0)
    parser.add_argument("--weight_scheduler", type=str, default="none",
                        choices=['none', 'square'])
    
    parser.add_argument("--na_weight", type=float, default=1.0)
    parser.add_argument("--pointer", type=str, default='pointer',
                        choices=['aligner', 'pointer', 'biaffine'])  
    parser.add_argument("--query", type=str, default='input',
                        choices=['input', 'attention'])
    parser.add_argument("--pre_norm", action="store_true")
    parser.add_argument("--no_linear_class", action="store_true")

    # v2 only
    parser.add_argument("--loss", type=str, default="ce",
                        choices=["ce", "focal", "ldam", "dice", "two"])
    parser.add_argument("--kl", type=str, default="none",
                        choices=['none', 'pq', 'qp', 'both'])
    parser.add_argument("--kl_alpha", type=float, default=0.0)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--ldam_max_m", type=float, default=0.5)
    parser.add_argument("--ldam_s", type=float, default=30)
    parser.add_argument("--dice_alpha", type=float, default=0.01)
    parser.add_argument("--dice_gamma", type=float, default=1.0)
    
    parser.add_argument("--use_context", action="store_true")
    parser.add_argument("--context_lstm", action="store_true")
    
    parser.add_argument("--negative_sampling", action="store_true") # used for span base
    parser.add_argument("--hard_neg_dist", type=int, default=3)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    
    parser.add_argument("--dp", type=float, default=0.2)
    parser.add_argument("--act", type=str, default="relu",
                        choices=["relu", "gelu"])
    
    # span_att
    parser.add_argument("--span_layer_count", type=int, default=2)
    parser.add_argument("--max_span_count", type=int, default=30)
    parser.add_argument("--share_parser", action="store_true")
    parser.add_argument("--unscale", action="store_true") # for transformer not tri-affine attention
    parser.add_argument("--scale", type=str, default="none",
                        choices=["none", "sqrt", "triv1", "triv2"]) # for tri-affine attention
    parser.add_argument("--init_std", type=float, default=2e-4)
    parser.add_argument("--layer_norm", action="store_true")

    # embedding enhancing
    parser.add_argument("--bert_before_lstm", action="store_true")
    parser.add_argument("--subword_aggr", type=str, default="first",
                        choices=['first', 'mean', 'max'])
    parser.add_argument("--bert_output", type=str, default="last",
                        choices=['last', 'concat-last-4', 'mean-last-4'])
    parser.add_argument("--reinit", type=int, default=0)
    
    parser.add_argument("--word", action="store_true")
    parser.add_argument("--word_embed", type=str, default="")
    parser.add_argument("--word_dp", type=float, default=0.5)
    parser.add_argument("--word_freeze", action="store_true")
    
    parser.add_argument("--char", action="store_true")
    parser.add_argument("--char_layer", type=int, default=1)
    parser.add_argument("--char_dim", type=int, default=50)
    parser.add_argument("--char_dp", type=float, default=0.2)
    
    parser.add_argument("--pos", action="store_true")
    parser.add_argument("--pos_dim", type=int, default=50)
    parser.add_argument("--pos_dp", type=float, default=0.2)
    
    parser.add_argument("--agg_layer", type=str, default="lstm",
                        choices=["lstm", "transformer"])
    parser.add_argument("--lstm_dim", type=int, default=1024)
    parser.add_argument("--lstm_layer", type=int, default=1)
    parser.add_argument("--lstm_dp", type=float, default=0.2)
    
    parser.add_argument("--tag", type=str, default="")
    
    return parser

def generate_loss_config(args):
    loss_dict = {'name':args.loss}
    if args.loss == "ce":
        loss_dict['na_weight'] = args.na_weight
    if args.loss == "focal":
        loss_dict['gamma'] = args.focal_gamma
        loss_dict['alpha'] = args.focal_alpha
    if args.loss == "ldam":
        from data_util import get_cls_num_list
        loss_dict['cls_num_list'] = get_cls_num_list(args.version)
        loss_dict['max_m'] = args.ldam_max_m
        loss_dict['s'] = args.ldam_s
    if args.loss == "dice":
        loss_dict['alpha'] = args.dice_alpha
        loss_dict['gamma'] = args.dice_gamma
    if args.loss == "two":
        loss_dict['na_weight'] = args.na_weight
    loss_dict['label_smoothing'] = args.label_smoothing
    
    if args.token_aux or args.model == "TokenModel":
        loss_dict['token_schema'] = args.token_schema
        if args.token_aux:
            loss_dict['token_aux_weight'] = args.token_aux_weight
    if args.negative_sampling:
        loss_dict['negative_sampling'] = True
        loss_dict['hard_neg_dist'] = args.hard_neg_dist
    loss_dict['class_loss_weight'] = args.class_loss_weight
    loss_dict['filter_loss_weight'] = args.filter_loss_weight
    loss_dict['dp'] = args.dp
    loss_dict['trans_aux'] = args.trans_aux
    if args.trans_aux:
        loss_dict['trans_aux_weight'] = args.trans_aux_weight
    else:
        loss_dict['trans_aux_weight'] = 0
        
    if args.kl != "none":
        loss_dict['kl'] = args.kl
        loss_dict['kl_alpha'] = args.kl_alpha
    return loss_dict
    
def generate_config(args):
    bert_config = {'bert_before_lstm':True if args.bert_before_lstm else False,
                   'subword_aggr':args.subword_aggr,
                   'bert_output':args.bert_output,
                   'reinit':args.reinit}
    if args.word:
        word_embedding_config = {'path':'',
                                'dropout':args.word_dp,
                                'dim':0,
                                'padding_idx':0,
                                'freeze':args.word_freeze}
        if args.version.find("ace04") >= 0:
            if not args.word_embed:
                word_embedding_config['path'] = 'data/ace04/wiki.npy'
                word_embedding_config['dim'] = 300
            elif args.word_embed == "glove":
                word_embedding_config['path'] = 'data/ace04/glove.npy'
                word_embedding_config['dim'] = 100
            elif args.word_embed == "cc":
                word_embedding_config['path'] = 'data/ace04/cc.npy'
                word_embedding_config['dim'] = 300                
            word_embedding_config['padding_idx'] = 15792
        if args.version.find("ace05") >= 0:
            if not args.word_embed:
                word_embedding_config['path'] = 'data/ace05/wiki.npy'
                word_embedding_config['dim'] = 300
            elif args.word_embed == "glove":
                word_embedding_config['path'] = 'data/ace05/glove.npy'
                word_embedding_config['dim'] = 100
            elif args.word_embed == "cc":
                word_embedding_config['path'] = 'data/ace05/cc.npy'
                word_embedding_config['dim'] = 300                
            word_embedding_config['padding_idx'] = 16061
        if args.version.find("genia91") >= 0:
            word_embedding_config['path'] = 'data/genia91/BioWordVec_PubMed_MIMICIII_d200.npy'
            word_embedding_config['dim'] = 200
            word_embedding_config['padding_idx'] = 25833
        if args.version.find("kbp") >= 0:
            if not args.word_embed or args.word_embed == "cc":
                word_embedding_config['path'] = 'data/kbp/cc.npy'
                word_embedding_config['dim'] = 300  
            word_embedding_config['padding_idx'] = 23228    
            
    else:
        word_embedding_config = {}
        
    if args.char:
        char_embedding_config = {'layer':args.char_layer,
                                'dropout':args.char_dp,
                                'dim':args.char_dim,
                                'padding_idx':0}
        if args.version.find("ace05") >= 0:
            char_embedding_config['padding_idx'] = 86
        if args.version.find("genia91") >= 0:
            char_embedding_config['padding_idx'] = 84
        if args.version.find("ace04") >= 0:
            char_embedding_config['padding_idx'] = 84
        if args.version.find("kbp") >= 0:
            char_embedding_config['padding_idx'] = 168
    else:
        char_embedding_config = {}
        
    if args.pos:
        pos_embedding_config = {'dropout':args.pos_dp,
                                'dim':args.pos_dim,
                                'padding_idx':0}
        if args.version.find("ace05") >= 0:
            pos_embedding_config['padding_idx'] = 45
        if args.version.find("ace04") >= 0:
            pos_embedding_config['padding_idx'] = 46
        if args.version.find("genia91") >= 0:
            pos_embedding_config['padding_idx'] = 1084
        if args.version.find("kbp") >= 0:
            pos_embedding_config['padding_idx'] = 46
    else:
        pos_embedding_config = {}
        
    lstm_config = {'name':args.agg_layer,
                   'dim':args.lstm_dim,
                   'layer':args.lstm_layer,
                   'dropout':args.lstm_dp,
                   'context_lstm':args.context_lstm}
    if args.lstm_layer == 1 and args.agg_layer == "lstm":
        lstm_config['dropout'] = 0.0
    if args.agg_layer == "transformer":
        lstm_config['dropout'] = 0.1 # use transformer default
        
    other_config = {'prenorm':args.pre_norm,
                    'span_layer_count':args.span_layer_count,
                    'max_span_count':args.max_span_count,
                    'share_parser':args.share_parser,
                    'unscale':args.unscale,
                    'act':args.act}
    
    return bert_config, \
        word_embedding_config, \
        char_embedding_config, \
        pos_embedding_config, \
        lstm_config, \
        other_config
