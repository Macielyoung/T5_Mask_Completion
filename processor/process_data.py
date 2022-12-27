from collections import defaultdict
from datasets import load_dataset
import random
random.seed(215)
from extractor import SketchExtractor
from zhconv import convert
sketch_extractor = SketchExtractor(model='lac')


# passage_dataset = load_dataset('beyond/chinese_clean_passages_80m', split='train[:128]')
passage_dataset = load_dataset('beyond/chinese_clean_passages_80m')
passage_dataset = passage_dataset['train'].filter(lambda example: len(example['passage']) <= 100)
print(passage_dataset)

N = len(passage_dataset)
k = 40000000
passage_dataset = passage_dataset.select(random.sample(range(N), k=k))
print("read data successfully")
# print(passage_dataset)

def add_sketch_to_dataset(examples):
    """
    """
    res = defaultdict(list)
    passages = examples['passage']
    for p in passages:
        # 针对部分繁体字，先做文字简写
        p = convert(p, 'zh-cn')
        # passage:
        res['text'].append(p)
        # _, kws = sketch_extractor.get_kws(p, top=max(len(jieba.lcut(p))//5,1))
        _, kws = sketch_extractor.get_kws(p, ratio=0.5)
        # we plan to use `fnlp/bart-large-chinese` for pre-training, the mask token is `[MASK]`
        sketch = sketch_extractor.get_sketch_from_kws(p, kws)
        res['sketch'].append(sketch)
    return res


dataset_with_sketch = passage_dataset.map(add_sketch_to_dataset, 
                                          batched=True, 
                                          batch_size=50, 
                                          num_proc=20,
                                          remove_columns=['passage'])
dataset_with_sketch = dataset_with_sketch.train_test_split(test_size=0.01)

print(dataset_with_sketch)

dataset_with_sketch.save_to_disk(f'../saved_datasets/chinese_4kw_sketches')
