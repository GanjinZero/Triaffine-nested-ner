import os
import ujson
import gensim
import fasttext
from gensim.test.utils import datapath
import numpy as np
from tqdm import tqdm


def get_vocab(file_path):
    with open(file_path, "r") as f:
        df = ujson.load(f)
    vocab_set = set()
    pos_set = set()
    for line in df:
        tokens = line['tokens']
        vocab_set.update(tokens)
        if 'ltokens' in line:
            ltokens = line['ltokens']
            vocab_set.update(ltokens)
        if 'rtokens' in line:
            rtokens = line['rtokens']
            vocab_set.update(rtokens)
        pos = line.get('pos',[])
        pos_set.update(pos)
    return list(vocab_set), list(pos_set)

def generate_vocab_embed(dataset, embedding=None, embedding_type="fasttext"):
    if dataset == "ace05":
        file_list = ["./data/ace05/ace05_train_context.json",
                     "./data/ace05/ace05_dev_context.json",
                     "./data/ace05/ace05_test_context.json"]
    if dataset == "ace04":
        file_list = ["./data/ace04/ace04_train_context.json",
                     "./data/ace04/ace04_dev_context.json",
                     "./data/ace04/ace04_test_context.json"]        
    if dataset == "genia91":
        file_list = ["./data/genia91/genia_train_dev_context.json",
                     "./data/genia91/genia_test_context.json"]
    if dataset == "kbp":
        file_list = ["./data/kbp/train_context.json",
                     "./data/kbp/dev_context.json",
                     "./data/kbp/test_context.json"]        
    # file_list = [f"data/{dataset}/{mode}_sample.json" for mode in ["train", "dev", "test"]]

    vocab = set()
    char_vocab = set()
    pos_vocab = set()
    for file in file_list:
        words, poss = get_vocab(file)
        vocab.update(words)
        pos_vocab.update(poss)
        
    for word in vocab:
        char_vocab.update(word)
        
    print(f'Vocab count: {len(vocab)}')
    print(f'Char vocab count: {len(char_vocab)}')
    print(f'POS vocab count: {len(pos_vocab)}')
    
    vocab = sorted(list(vocab))
    vocab.extend(['[UNK]', '[PAD]'])
    char_vocab = sorted(list(char_vocab))
    char_vocab.extend(['[UNK]', '[PAD]'])
    pos_vocab = sorted(list(pos_vocab))
    pos_vocab.extend(['[UNK]', '[PAD]'])
    
    with open(f"./data/{dataset}/word2id.json", "w", encoding="utf-8") as f:
        ujson.dump({word: idx for idx, word in enumerate(vocab)}, f, indent=2)
    with open(f"./data/{dataset}/char2id.json", "w", encoding="utf-8") as f:
        ujson.dump({char: idx for idx, char in enumerate(char_vocab)}, f, indent=2)
    with open(f"./data/{dataset}/pos2id.json", "w", encoding="utf-8") as f:
        ujson.dump({pos: idx for idx, pos in enumerate(pos_vocab)}, f, indent=2)
        
    if embedding_type == "fasttext":
        model = fasttext.load_model(embedding)
    if embedding_type == "txt":
        model = load_txt_embed(embedding)
    # if embedding_type == "model":
    #     model = gensim.models.Word2Vec.load(embedding)
    # if embedding_type == "bin":
    #     model = gensim.models.KeyedVectors.load_word2vec_format(embedding, binary=True)
    
    if embedding_type == "fasttext":
        word_embed = [model.get_word_vector(word) for word in tqdm(vocab[0:-2])]
        word_embed.append(np.random.randn(len(word_embed[0])))
        word_embed.append(np.zeros_like(word_embed[0]))
    if embedding_type == "txt":
        dim = len(model['the'])
        word_embed = [model.get(word, model.get(word.lower(), np.random.randn(dim))) for word in tqdm(vocab[0:-2])]
        word_embed.append(np.random.randn(len(word_embed[0])))
        word_embed.append(np.zeros_like(word_embed[0]))
    
    embedding_name = embedding.split('/')[-1].split('.')[0]
    with open(f"./data/{dataset}/{embedding_name}.npy", "wb") as ff:
        np.save(ff, np.array(word_embed))
        
def load_txt_embed(embedding_path):
    model = {}
    with open(embedding_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        sp_line = line.split()
        if len(sp_line) == 2:
            continue
        name = sp_line[0]
        emb = [float(x) for x in sp_line[1:]]
        model[name] = np.array(emb)
    return model
       
 
if __name__ == "__main__":
    generate_vocab_embed('kbp', '../pretraining-models/cc.en.300.bin', 'fasttext')
    generate_vocab_embed('ace04', '../pretraining-models/cc.en.300.bin', 'fasttext')
    generate_vocab_embed('ace05', '../pretraining-models/cc.en.300.bin', 'fasttext')
    generate_vocab_embed('genia91', '../pretraining-models/BioWordVec_PubMed_MIMICIII_d200.bin', 'fasttext')
