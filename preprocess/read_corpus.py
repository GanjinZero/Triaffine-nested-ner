import os
import re

def read_single_file(file_path):
    
    if not file_path.endswith('sgm'):
        return ""
    
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    
    pre = re.compile('>(.*?)<')
    filtered_text = ' '.join(pre.findall(" ".join(lines)))
    
    return filtered_text

def read_corpus(folder):
    result = {}
    # print(folder)
    file_list = os.listdir(folder)
    for file in file_list:
        cur_path = os.path.join(folder, file)
        if file.endswith('sgm'):
            result[cur_path] = read_single_file(cur_path)
        elif os.path.isdir(cur_path):
            new_result = read_corpus(cur_path)
            for key, value in new_result.items():
                result[key] = value
    return result
    

if __name__ == "__main__":
    # txt = read_single_file('ace_2004/data/English/arabic_treebank/20000715_AFP_ARB.0054.eng.sgm')
    # result = read_corpus('ace_2004/data/English')
    result = read_corpus('ace_2005_td_v7/data/English')
    import ipdb; ipdb.set_trace()
