import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from data_util import NestedNERDataset, my_collate_fn
from tensorboardX import SummaryWriter
from tqdm import tqdm
import shutil
import json
import ipdb
import sys
import numpy as np
from evaluation import decode, metric, write_predict
from train_parser import generate_parser, generate_config, generate_loss_config
from train_utils import generate_output_folder_name, generate_optimizer_scheduler
from model.span import SpanModel
from model.span_att_v2 import SpanAttModelV2, SpanAttModelV3
from input_util import prepare_input
from train_utils import main_name, weight_scheduler
import random
from torch_ema import ExponentialMovingAverage


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def run(args):
    if args.seed != -1:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed) 

    output_basename = generate_output_folder_name(args)
    print(output_basename)
    writer = SummaryWriter(comment=output_basename[0:200])
    output_path = os.path.join(args.output_base_dir, output_basename)

    if os.path.exists(f'{output_path}/metric_log'):
        print('Metric Log exists.')
        sys.exit()

    try:
        os.system(f"mkdir -p {output_path}")
    except BaseException:
        pass

    try:
        os.system(f"rm -rf {output_path}/metric_log")
    except BaseException:
        pass
    
    if args.model in ["SpanModel"]:
        args.schema = "span"
    if args.model in ["DETR"]:
        args.schema = "DETR"
    if args.model in ["DETRSeq"]:
        args.schema = "DETRSeq"
    if args.model in ["OneStageSpan", "TwoStageSpan"]:
        args.schema = "softspan"
        
    if args.freeze_bert:
        if args.use_context:
            con = "_context"
        else:
            con = ""
        train_bert_embed = f'./data/{args.version}/{main_name(args.bert_name_or_path)}_train_{args.truncate_length}{con}.hdf5'
        dev_bert_embed = f'./data/{args.version}/{main_name(args.bert_name_or_path)}_dev_{args.truncate_length}{con}.hdf5'
        test_bert_embed = f'./data/{args.version}/{main_name(args.bert_name_or_path)}_test_{args.truncate_length}{con}.hdf5'
    else:
        train_bert_embed = None
        dev_bert_embed = None
        test_bert_embed = None
    
    train_dataset = NestedNERDataset(args.version, 'train', args.bert_name_or_path, args.truncate_length, args.schema, args.use_context, args.token_schema, args.soft_iou, bert_embed=train_bert_embed)
    dev_dataset = NestedNERDataset(args.version, 'dev', args.bert_name_or_path, args.truncate_length, args.schema, args.use_context, args.token_schema, args.soft_iou, bert_embed=dev_bert_embed)
    test_dataset = NestedNERDataset(args.version, 'test', args.bert_name_or_path, args.truncate_length, args.schema, args.use_context, args.token_schema, args.soft_iou, bert_embed=test_bert_embed)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=my_collate_fn, shuffle=True, num_workers=1)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=my_collate_fn, shuffle=False, num_workers=1)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=my_collate_fn, shuffle=False, num_workers=1)

    encoder_config_dict = generate_config(args)
    loss_config_dict = generate_loss_config(args)
    
    score_setting = {args.score:True}
    if args.no_linear_class:
        score_setting['no_linear_class'] = True
    if args.type_attention:
        score_setting['type_attention'] = True
    score_setting['dp'] = args.dp
    score_setting['att_dim'] = args.att_dim
    score_setting['no_tri_mask'] = args.no_tri_mask
    score_setting['reduce_last'] = args.reduce_last
    score_setting['scale'] = args.scale
    score_setting['init_std'] = args.init_std
    score_setting['layer_norm'] = args.layer_norm
    score_setting['rel_pos_attn'] = args.rel_pos_attn
    score_setting['rel_pos'] = args.rel_pos
    score_setting['rel_k'] = args.rel_k
        

    if args.model == "SpanModel":
        model = SpanModel(args.bert_name_or_path, encoder_config_dict,
                          len(train_dataset.type2id), score_setting,
                          loss_config=loss_config_dict).to(args.device)
    if args.model == "SpanAttModelV3":
        model = SpanAttModelV3(args.bert_name_or_path, encoder_config_dict,
                          len(train_dataset.type2id), score_setting,
                          loss_config=loss_config_dict).to(args.device)
    optimizer, scheduler = generate_optimizer_scheduler(args, model, len(train_dataloader))
    
    if args.ema > 0.:
        ema = ExponentialMovingAverage(model.parameters(), decay=args.ema)
    else:
        ema = None
    
    steps = 0
    best_dev_metric = None
    best_test_metric = None
    early_stop_count = 0
    best_epoch_idx = 0

    for epoch_idx in range(1, args.train_epoch + 1):
        epoch_dev_metric, epoch_test_metric, steps = train_one_epoch(model, steps, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, writer, args, epoch_idx, ema)
        
        print('Dev_Epoch' + str(epoch_idx), epoch_dev_metric)
        print('Test_Epoch' + str(epoch_idx), epoch_test_metric)
        with open(os.path.join(output_path, 'metric_log'), 'a+', encoding='utf-8') as f:
            f.write('---\n')
            f.write('Dev_Epoch' + str(epoch_idx) + ' ' + str(epoch_dev_metric) + "\n")
            f.write('Test_Epoch' + str(epoch_idx) + ' ' + str(epoch_test_metric) + "\n")
            
        # Early Stop
        if best_dev_metric is None:
            best_dev_metric = epoch_dev_metric
            best_test_metric = epoch_test_metric
            best_epoch_idx = epoch_idx
            torch.save(model, os.path.join(output_path, f"epoch{epoch_idx}.pth"))
            if ema is not None:
                torch.save(model, os.path.join(output_path, f"ema{epoch_idx}.pth"))
        else:
            if epoch_dev_metric['f1'] >= best_dev_metric['f1']:
                best_dev_metric = epoch_dev_metric
                best_test_metric = epoch_test_metric
                best_epoch_idx = epoch_idx
                early_stop_count = 0
                torch.save(model, os.path.join(output_path, f"epoch{epoch_idx}.pth"))
                if ema is not None:
                    torch.save(model, os.path.join(output_path, f"ema{epoch_idx}.pth"))
            else:
                if args.save_every_epoch:
                    torch.save(model, os.path.join(output_path, f"epoch{epoch_idx}.pth"))
                    if ema is not None:
                        torch.save(model, os.path.join(output_path, f"ema{epoch_idx}.pth"))
                early_stop_count += 1

        if args.early_stop_epoch > 0 and early_stop_count >= args.early_stop_epoch:
            print(f"Early Stop at Epoch {epoch_idx}, \
                    F1 does not improve on dev set for {early_stop_count} epoch.")
            break

    print('Best_Dev_Epoch' + str(best_epoch_idx), best_dev_metric)
    print('Best_Test_Epoch' + str(best_epoch_idx), best_test_metric)
    with open(os.path.join(output_path, 'metric_log'), 'a+', encoding='utf-8') as f:
        f.write('---\n')
        f.write('Best_Dev_Epoch' + str(best_epoch_idx) + ' ' + str(best_dev_metric) + "\n")
        f.write('Best_Test_Epoch' + str(best_epoch_idx) + ' ' + str(best_test_metric) + "\n")
        
    best_path = os.path.join(output_path, f"epoch{best_epoch_idx}.pth")
    best_ema = os.path.join(output_path, f"ema{best_epoch_idx}.pth")
    new_path = os.path.join(output_path, "best_epoch.pth")
    new_ema_path = os.path.join(output_path, "best_ema.pth")
    os.system(f'cp {best_path} {new_path}')
    if ema is not None:
        os.system(f'cp {best_ema} {new_ema_path}')
    
    # predict using best epoch
    print('Predict dev and test dataset using best checkpoint')
    model = torch.load(best_path).to(args.device)
    if ema is not None:
        ema = torch.load(best_ema).to(args.device)
    dev_strict, dev_relax = decode(dev_dataloader, model, args, ema)
    test_strict, test_relax = decode(test_dataloader, model, args, ema)
    
    write_predict(dev_strict, os.path.join(output_path, 'dev_predict.txt'))
    write_predict(test_strict, os.path.join(output_path, 'test_predict.txt'))


def train_one_epoch(model, steps, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, writer, args, epoch_idx, ema):
    output_basename = generate_output_folder_name(args)
    output_path = os.path.join(args.output_base_dir, output_basename) 
    
    model.train()
    epoch_loss = 0.
    
    if args.weight_scheduler != "none":
        if hasattr(model, 'class_loss_weight'):
            w = weight_scheduler(epoch_idx, args=args, method=args.weight_scheduler)
            print(f'Set class loss weight {epoch_idx}/{args.train_epoch}:{w}')
            model.class_loss_weight = w
        
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", ascii=True)
    for batch_idx, batch in enumerate(epoch_iterator):
        inputs = prepare_input(batch, args)
        loss = model(**inputs)
            
        batch_loss = float(loss.item())
        epoch_loss += batch_loss

        writer.add_scalar('batch_loss', batch_loss)
                
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()
        epoch_iterator.set_description("Epoch_loss: %0.4f, Batch_loss: %0.4f" % (epoch_loss / (batch_idx + 1), batch_loss))

        if (steps + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if ema is not None:
                ema.update()
            model.zero_grad()
        steps += 1
        
    dev_strict, dev_relax = decode(dev_dataloader, model, args, ema)
    write_predict(dev_strict, os.path.join(output_path, f'dev_{epoch_idx}_predict.txt'))
    test_strict, test_relax = decode(test_dataloader, model, args, ema)
    write_predict(test_strict, os.path.join(output_path, f'test_{epoch_idx}_predict.txt'))
    dev_metric = metric(dev_dataloader.dataset, dev_strict)
    if dev_relax:
        dev_metric = {**dev_metric, **metric(dev_dataloader.dataset, dev_relax, "relax")}
        write_predict(dev_relax, os.path.join(output_path, f'dev_{epoch_idx}_relax_predict.txt'))
    test_metric = metric(test_dataloader.dataset, test_strict)
    if test_relax:
        test_metric = {**test_metric, **metric(test_dataloader.dataset, test_relax, "relax")}
        write_predict(test_relax, os.path.join(output_path, f'test_{epoch_idx}_relax_predict.txt'))

    for key in dev_metric:
        writer.add_scalar('dev_' + key, dev_metric[key])
    for key in test_metric:
        writer.add_scalar('test_' + key, test_metric[key])

    return dev_metric, test_metric, steps


def main():
    parser = generate_parser()
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
