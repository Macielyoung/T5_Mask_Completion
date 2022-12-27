import datasets
from transformers import AutoTokenizer
import numpy as np


def count_dataset(dataset, tokenizer, flag='train'):
    indicators = {}
    passage_lengths, sketch_lengths = [], []
    data_num = len(dataset)
    rid = 0
    for row in dataset:
        passage = row['passage']
        sketch = row['sketch']
        
        passage_tokens = tokenizer.tokenize(passage)
        sketch_tokens = tokenizer.tokenize(sketch)
        if rid % 1000 == 0:
            print("{} dataset proceed data row: {} / {}".format(flag, rid, data_num), flush=True)
            print("passage: {}, sketch: {}".format(passage, sketch), flush=True)
        
        passage_lengths.append(len(passage_tokens))
        sketch_lengths.append(len(sketch_tokens))
        rid += 1
        
    indicators['max_passage'] = max(passage_lengths)
    indicators['min_passage'] = min(passage_lengths)
    indicators['median_passage'] = np.median(passage_lengths)
    indicators['mean_passage'] = np.mean(passage_lengths)
    
    indicators['max_sketch'] = max(sketch_lengths)
    indicators['min_sketch'] = min(sketch_lengths)
    indicators['median_sketch'] = np.median(sketch_lengths)
    indicators['mean_sketch'] = np.mean(sketch_lengths)
    return indicators

saved_path = "../saved_datasets/chinese_clean_passages_80m_with_sketch"
sketch_dataset = datasets.load_from_disk(saved_path)
print("read data done!", flush=True)
print(sketch_dataset)
exit(0)

sketch_datase = sketch_dataset['train'].train_test_split(test_size=0.1)
print("split dataset to train and eval dataset", flush=True)

pretrained_model = "Langboat/mengzi-t5-base"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
tokenizer.add_tokens(["[mask]"])
print("load tokenizer done!", flush=True)

train_dataset = sketch_dataset['train']
eval_dataset = sketch_dataset['test']

train_indicators = count_dataset(train_dataset, tokenizer, flag='train')
print(train_indicators)

eval_indicators = count_dataset(eval_dataset, tokenizer, flag='eval')
print(eval_indicators)
