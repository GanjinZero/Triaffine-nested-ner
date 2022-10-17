import ujson
import os
from tqdm import tqdm
import copy


def align(input_path, output_path):
    with open(input_path, 'r') as f:
        df = ujson.load(f)
    
    pos_dict = {}
    for now_df in df:
        pos_dict["|||||".join(now_df['tokens'])] = now_df['pos']
        
    for now_df in df:
        now_df['lpos'] = pos_dict.get("|||||".join(now_df['ltokens']), [])
        now_df['rpos'] = pos_dict.get("|||||".join(now_df['rtokens']), [])
    
    with open(output_path, 'w') as f:
        ujson.dump(df, f, indent=2)
    
def main():
    folder = "../nested_v2/data/genia91/"
    align(os.path.join(folder, 'genia_train_dev_context.json'), './res/genia91_train_v4.json')
    align(os.path.join(folder, 'genia_test_context.json'), './res/genia91_test_v4.json')
    
if __name__ == "__main__":
    main()
