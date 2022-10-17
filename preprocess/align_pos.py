import os
import nltk
import ujson
from tqdm import tqdm


def align_pos(input_path, output_path):
    with open(input_path, "r") as f:
        df = ujson.load(f)
        
    for i, now_df in tqdm(enumerate(df)):
        df[i]['pos'] = [x[-1] for x in nltk.pos_tag(now_df['tokens'])]
        
    with open(output_path, "w") as f:
        ujson.dump(df, f, indent=2)
    
    
def main():
    align_pos("res/ace04_dev_v2.json", "res/ace04_dev_v3.json")
    align_pos("res/ace04_test_v2.json", "res/ace04_test_v3.json")
    align_pos("res/ace04_train_v2.json", "res/ace04_train_v3.json")
    
if __name__ == "__main__":
    main()