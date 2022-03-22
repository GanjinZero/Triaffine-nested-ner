import torch


def neg_step(label, w=None):
    if w is None:
        seq = label.size(1)
        w = torch.tril(torch.ones(seq, seq).to(
            label.device), diagonal=1).bool()
        w = torch.bitwise_and(w, w.t()).float()
    hard_negative = (torch.matmul(w.unsqueeze(0), label.float()) +
                     torch.matmul(label.float(), w.unsqueeze(0))).bool()
    return hard_negative


def negative_sampling(label, hard_neg_dist=2):
    # label: Batch * max_word * max_word
    sampling_label = label.clone()

    na_label = torch.max(label.reshape(-1))
    positive_label = torch.bitwise_and(label != na_label, label >= 0)
    positive_count = positive_label.sum(dim=-1).sum(dim=-1)
    negative_label = label == na_label

    if hard_neg_dist > 0:
        seq = label.size(1)
        w = torch.tril(torch.ones(seq, seq).to(
            label.device), diagonal=1).bool()
        w = torch.bitwise_and(w, w.t()).float()

    hard_negative = positive_label
    for _ in range(hard_neg_dist):
        hard_negative = neg_step(hard_negative, w)

    easy_negative = torch.bitwise_and(negative_label, ~hard_negative)
    random_num = torch.rand(easy_negative.size()).to(label.device)
    sample_res = random_num > (
        positive_count / (1e-6 + easy_negative.sum(dim=-1).sum(dim=-1))).unsqueeze(-1).unsqueeze(-1)
    sample_easy_negative = torch.bitwise_and(easy_negative, sample_res)
    sampling_label[sample_easy_negative] = -100
    return sampling_label


def iou(a, b):
    iou = 0
    if max(a[0], b[0]) < 1 + min(a[1], b[1]):
        iou = (1 + min(a[1], b[1]) - max(a[0], b[0])) / \
              (1 + max(a[1], b[1]) - min(a[0], b[0]))
    return iou

def tensor_idx_add(src, idx, value=1, coef=None):
    '''
    src: any shape
    idx: n * len(src.size())
    '''
    sz = src.size()
    if coef is None:
        coef = [1]
        for s in sz[::-1]:
            coef.append(s * coef[-1])
        coef = torch.LongTensor(coef[0:-1][::-1]).to(src.device).unsqueeze(0) # src.size() # .repeat(idx.size(0),1)
    new_src = (idx * coef).sum(-1)
    select_src = src.reshape(-1)
    select_src[new_src] += value
    return src.reshape(sz)

