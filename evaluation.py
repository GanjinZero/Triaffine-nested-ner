import torch
import os
from input_util import prepare_input
from tqdm import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def tint(t):
    if isinstance(t, torch.Tensor):
        return t.item()
    return t

def decode(dataloader, model, args, ema=None):
    # if accelerator is not None:
    #     return decode_accl(dataloader, model, args, accelerator)
    
    model.eval()
    predict_labels = []
    
    if ema is None:
        with torch.no_grad():
            for batch in dataloader:
                inputs = prepare_input(batch, args, train=False)
                result = model.predict(**inputs)

                for i in range(batch[0].size(0)):
                    predict_label = list(set([f'{tint(l[0])},{tint(l[1]) + 1} {dataloader.dataset.id2type[tint(l[2])]}' for l in result[i]]))
                    predict_labels.append(predict_label)
    else:
        with torch.no_grad():
            with ema.average_parameters():
                for batch in dataloader:
                    inputs = prepare_input(batch, args, train=False)
                    result = model.predict(**inputs)

                    for i in range(batch[0].size(0)):
                        predict_label = list(set([f'{tint(l[0])},{tint(l[1]) + 1} {dataloader.dataset.id2type[tint(l[2])]}' for l in result[i]]))
                        predict_labels.append(predict_label)
    
    return predict_labels, []

def decode_threshold(dataloader, model, args=None, accelerator=None, threshold_list=[0.5], device=None):
    if accelerator is not None:
        raise NotImplementedError
    
    if isinstance(threshold_list, float):
        threshold_list = [threshold_list]
    
    model.eval()
    predict_labels = [[] for _ in threshold_list]
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs = prepare_input(batch, args, train=False, device=device)
            result = model.predict(**inputs)

            for i in range(batch[0].size(0)):
                for idx, threshold in enumerate(threshold_list):
                    predict_label = list(set([f'{tint(l[0])},{tint(l[1]) + 1} {dataloader.dataset.id2type[tint(l[2])]}' for l in result[i] if l[3] >= threshold]))
                    predict_labels[idx].append(predict_label)
    
    return predict_labels, []

def write_predict(predict_labels, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for predcit_label in predict_labels:
            f.write('|'.join(predcit_label) + '\n')
    
    
def metric(dataset, predict_labels_0, suffix=None):
    correct = 0
    predict_count = 0
    label_count = 0
    
    true_labels = []
    for i in range(len(dataset)):
        true_label = []
        for ent in dataset.df[i]['entities']:
            st = ent['start']
            ed = ent['end']
            tp = ent['type']
            true_label.append(f'{st},{ed} {tp}')
        true_labels.extend(true_label)
        predict_label = predict_labels_0[i]
        
        for l in predict_label:
            if l in true_label:
                correct += 1
        predict_count += len(predict_label)
        label_count += len(true_label)
    if correct == 0:
        p = 0
        r = 0
    else:
        p = correct / predict_count
        r = correct / label_count
    if p == 0 or r == 0:
        f1 = 0
    else:
        f1 = 2 * p * r / (p + r)
    
    if suffix is None:
        return {'p':p, 'r':r, 'f1':f1}
    return {f'{suffix}_p':p, f'{suffix}_r':r, f'{suffix}_f1':f1}

