import ujson
import os
from read_corpus import read_corpus
from tqdm import tqdm
import copy
import nltk


def clean(sth):
    f = sth.replace('-LCB-', "").replace('-RCB-', "").replace('{','').replace('}','')
    f = f.replace('-LRB-', "(").replace('-RRB-',")").replace(".","").replace('`','').replace('\'','').replace('"','').replace('-LSB-','[').replace('-RSB-',']').replace('<UNK>','')
    f = f.replace('&amp;T', "").replace('&amp;M', "").replace('&amp;A', '').replace('&amp;', "").replace('&AMP;', '').replace('\t','').replace('\n','').replace('&','').replace('-','').replace('M','').replace('A','')
    return f

def remove_space(sth):
    if isinstance(sth, str):
        return clean("".join([w for w in sth.split() if w]))
    elif isinstance(sth, list):
        return clean("".join([w for w in sth]))
    elif isinstance(sth, dict):
        new_sth = copy.deepcopy(sth)
        for key, value in new_sth.items():
            new_sth[key] = clean("".join([w for w in value.split() if w]))
        for value in new_sth.values():
            assert value.find('\'') == -1
            assert value.find('`') == -1
        
        return new_sth
    return ""

def match(cop, tok, con_tok):
    for key, value in cop.items():
        if value.find(con_tok) >= 0:
            return key
    for key, value in cop.items():
        if value.find(tok) >= 0:
            return key
    return ""

def get_idx(tokens, ltokens, rtokens, txt):
    l_idx = 0
    r_idx = 0
    return l_idx, r_idx

def genearte(txt, l_idx, r_idx):
    l_tokens = txt[0:l_idx]
    r_tokens = txt[r_idx:]
    return [w for w in l_tokens.split() if w], [w for w in r_tokens.split() if w]

def j(lid, df, attr):
    tks = []
    for id in lid:
        tks.extend(df[id][attr])
    return tks

def align(input_path, output_path, version):
    print(input_path)
    print(output_path)
    corpus = read_corpus(version)
    remove_space_corpus = remove_space(corpus)

    with open(input_path, "r") as f:
        df = ujson.load(f)
        
    new_df = []
    find_count = 0
    sen_dict = {}
    
    for i, now_df in enumerate(df):
        if 'pos' not in df[i]:
            df[i]['pos'] = [x[-1] for x in nltk.pos_tag(now_df['tokens'])]
    
    for i, now_df in enumerate(df):
        now_df = df[i]
        
        # modify ltokens and rtokens
        tokens = now_df['tokens']
        if 'ltokens' in now_df:
            ltokens = now_df['ltokens']
        else:
            ltokens = []
        if 'rtokens' in now_df:
            rtokens = now_df['rtokens']
        else:
            rtokens = []        
        remove_space_tokens = remove_space(tokens)
        remove_space_context = remove_space(ltokens + tokens + rtokens)
        
        corpus_name = match(remove_space_corpus, remove_space_tokens, remove_space_context)
        if corpus_name:
            find_count += 1
            sen_dict[i] = corpus_name
        else:
            print(tokens)
            print(remove_space_tokens)
        
        new_df.append(now_df)
        
    corpus_list = set(sen_dict.values())
    for corpus_name in corpus_list:
        id_list = []
        for i, now_df in tqdm(enumerate(df)):
            if sen_dict[i] == corpus_name:
                id_list.append(i)
        
        print(corpus_name, id_list)
        for idx, i in enumerate(id_list):
            li = id_list[0:idx]
            ri = id_list[idx+1:]
            new_df[i]['ltokens'] = j(li, new_df, 'tokens')
            new_df[i]['rtokens'] = j(ri, new_df, 'tokens')
            new_df[i]['lpos'] = j(li, new_df, 'pos')
            new_df[i]['rpos'] = j(ri, new_df, 'pos')
    
        
    print(f'Find count:{find_count}/{len(df)}')
    print('---')
        
    with open(output_path, "w") as f:
        ujson.dump(new_df, f, indent=2)
        
    return
     
def main():
    ace04 = 'ace_2004/data/English'
    ace05 = 'ace_2005_td_v7/data/English'
    align('../nested_v2/data/ace05/ace05_train_context.json', 'res/ace05_train_v4.json', ace05)
    align('../nested_v2/data/ace05/ace05_dev_context.json', 'res/ace05_dev_v4.json', ace05)
    align('../nested_v2/data/ace05/ace05_test_context.json', 'res/ace05_test_v4.json', ace05)
    align('../nested_v2/data/ace04/ace04_train.json', 'res/ace04_train_v4.json', ace04)
    align('../nested_v2/data/ace04/ace04_dev.json', 'res/ace04_dev_v4.json', ace04)
    align('../nested_v2/data/ace04/ace04_test.json', 'res/ace04_test_v4.json', ace04)

if __name__ == "__main__":
    main()
