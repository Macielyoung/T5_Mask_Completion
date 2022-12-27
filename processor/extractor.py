import jieba.analyse as analyse
import re
from LAC import LAC
import math

###################################################### 
# get sketch for pre-training
###################################################### 


table = str.maketrans({"-":  r"\-", "]":  r"\]", "[":  r"\[", "\\": r"\\", \
                       "^":  r"\^", "$":  r"\$", "*":  r"\*", ".":  r"\.", \
                        "(":  r"\(", ")":  r"\)", \
                       })

class SketchExtractor:
    def __init__(self, model='lac'):
        assert model in ['lac','jieba'], '`model` only support `yake`, `bert` or `jieba`'
        self.model = model
        self.mask = '<mask>'
        self.sep = ' '
        if model == 'jieba': # for Chinese
            print('You are using Chinese version.\n --mask token: "[MASK]"\n--sep:""')
            self.extractor = analyse
            self.mask = '[mask]'
            self.sep = ''
        if model == 'lac':
            print("Your are user Baidu LAC model to get importance keywords!")
            self.extractor = LAC(mode='rank')
            self.mask = '[mask]'
            self.sep = ''
            

    def get_kws(self, s, max_ngram=3, top=10, ratio=0.6, aspect_keywords=None, use_aspect_as_doc_embedding=False):
        if self.model == 'jieba':
            kws = self.extractor.extract_tags(s, topK=top)
            return [], kws
        if self.model == "lac":
            lac_results = self.extractor.run(s)
            # print(lac_results)
            lac_tokens, _, lac_levels = lac_results
            lac_index_dict = {index: level for index, level in enumerate(lac_levels) if level >= 2}
            # print(lac_index_dict)
            sorted_lac_index_items = sorted(lac_index_dict.items(), key=lambda x: x[1], reverse=True)
            sorted_lac_indices = [pair[0] for pair in sorted_lac_index_items]
            # print(sorted_lac_indices)
            # lac_indices = [index for index, level in enumerate(lac_levels) if level >= 2]
            select_num = math.ceil(len(sorted_lac_indices) * ratio)
            select_indices = sorted_lac_indices[:select_num]
            kws = [lac_tokens[index] for index in select_indices]
            return [], kws
        return [], []

    def get_sketch_from_kws(self, s, kws, template=4):
        """
        TODO: keywords extracted by YAKE may not always be the same as original, like "IBM's" will be "IBM".
              for template 3/4, a workaround is split keywords into single words, then match
        template:
        1 --> keywords ordered by importance (given by the extractor), joint by a single space 
        2 --> keywords ordered by the original order in `s`, joint by a single space
        3 --> keywords ordered by the original order and frequences in `s`, joint by a single space
        4 --> same as above, but joint by a single <mask> token (the default GENIUS mode)
        """
        mask = self.mask
        sep = self.sep
        if template == 1:
            return ' '.join(kws)
        if template == 2:
            orders = []
            remain_kws = []
            for w in kws:
                # yake will ommit some tokens such as `'s` in a phrase, 
                # resulting in mismatch in the original text
                # in this situation, currently we choose to simply jump over...
                try:
                    order = s.index(w)
                    orders.append(order)
                    remain_kws.append(w)
                except:
                    # print(w, 'not match')
                    pass
            kws = remain_kws
            kws_with_order = [(w,o) for w,o in zip(kws, orders)]
            kws_with_order = sorted(kws_with_order, key=lambda x:x[1])
            osrted_kws = [p[0] for p in kws_with_order]
            return ' '.join(osrted_kws)
        if template == 3:
            all_ids = []
            for w in kws: # get the position for each work
                try:
                    for m in list(re.finditer(re.escape(w.translate(table)),s)): 
                        all_ids += list(range(m.start(),m.end()))
                except Exception as e:
                    print(e)
                    print(w, ' |not found in| ', s)
            all_ids = sorted(list(set(all_ids)))
            # fill with mask token for discontinuous position
            masked_text = []
            for i,id in enumerate(all_ids):
                if id - all_ids[i-1] > 1: # something in between
                    masked_text.append(' ')
                masked_text.append(s[id])
            return ''.join(masked_text)
        if template == 4:
            all_ids = []
            for w in kws: # get the position for each work
                try:
                    for m in list(re.finditer(re.escape(w.translate(table)),s)): 
                        all_ids += list(range(m.start(),m.end()))
                except Exception as e:
                    print(e)
                    print(w, ' |not found in| ', s)
            all_ids = sorted(list(set(all_ids)))
            # fill with mask token for discontinuous position
            masked_text = []
            for i,id in enumerate(all_ids):
                if i == 0 and id != 0: # mask for the begining
                    masked_text.append(f'{mask}{sep}')
                if sep == ' ' and id - all_ids[i-1] == 2 and s[id-1] == ' ': # a space in between
                    masked_text.append(' ')
                if sep == '' and id - all_ids[i-1] == 2:
                    masked_text.append(f'{sep}{mask}{sep}')
                if id - all_ids[i-1] > 2: # something in between
                    masked_text.append(f'{sep}{mask}{sep}')
                masked_text.append(s[id])
                if i == len(all_ids)-1 and id != len(s)-1: # mask for the end
                    masked_text.append(f'{sep}{mask}')
            return ''.join(masked_text)
    
    def get_sketch(self, s, max_ngram=3, top=10, ratio=0.8, aspect_keywords=None, use_aspect_as_doc_embedding=False, template=4):
        _, kws = self.get_kws(s, max_ngram, top, ratio, aspect_keywords, use_aspect_as_doc_embedding)
        sketch = self.get_sketch_from_kws(s, kws, template=template)
        return sketch
              
               
"""
Example:
E = SketchExtractor(model='yake')
s = '''The purpose of the AAAI conference series is to promote research in Artificial Intelligence (AI) and foster scientific exchange between researchers, practitioners, scientists, students, and engineers across the entirety of AI and its affiliated disciplines. '''
E.get_sketch(s, top=7, template=4)
E.get_kws(s, top=7)

model='yake'
template = 1:
'AAAI conference series foster scientific exchange Artificial Intelligence AAAI conference research in Artificial exchange between researchers affiliated disciplines'
template = 2:
'AAAI conference series AAAI conference research in Artificial Artificial Intelligence foster scientific exchange exchange between researchers affiliated disciplines'
template = 3:
'AAAI conference series research in Artificial Intelligence foster scientific exchange between researchers affiliated disciplines'
template = 4:
'<mask> AAAI conference series <mask> research in Artificial Intelligence <mask> foster scientific exchange between researchers <mask> affiliated disciplines <mask>'
"""


###################################################### 
# Basic Text Cleaning 
###################################################### 

def remove_special_characters(text):
    # only remain alphabets, digits, and main punctuations
    pat = r'[^a-zA-z0-9.,!?/:;\"\'\s]' 
    new_text =  re.sub(pat, '', text)
    return new_text
def remove_brakets(text):
    # remove [...],(...)
    text =  re.sub(r'\[(.*)\]', '', text)
    text =  re.sub(r'\((.*)\)', '', text)
    return text
def clean_pipeline(text):
    return remove_brakets(remove_special_characters(text))


import torch, random
import numpy as np
def setup_seed(seed):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


from torch.utils.data import Dataset
class List2Dataset(Dataset):
    def __init__(self, inputs):
        # inputs: list of strings
        # this class is for huggingface pipeline batch inference
        self.inputs = inputs
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, i):
        return self.inputs[i]


if __name__ == "__main__":
    # sketch_extractor = SketchExtractor(model='jieba')
    # s = "今天天气真的不错，出去散步了好长时间，顺便买了一些栗子吃吃。"
    # _, kws = sketch_extractor.get_kws(s)
    # print(kws)
    # sketch = sketch_extractor.get_sketch_from_kws(s, kws)
    # print(sketch)
    
    sketch_extractor = SketchExtractor(model='lac')
    s = "今天天气真的不错，出去散步了好长时间，顺便买了一些栗子吃吃。"
    _, kws = sketch_extractor.get_kws(s, ratio=0.6)
    print(kws)
    # sketch = sketch_extractor.get_sketch_from_kws(s, kws)
    # print(sketch)