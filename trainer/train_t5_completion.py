from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq, TrainingArguments, Trainer, Seq2SeqTrainingArguments
from datasets import load_metric, load_from_disk
import os
import numpy as np
import nltk
from loguru import logger
from zhconv import convert
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def preprocess_function(examples):
    # T5 model
    source_max_length = 320
    target_max_length = 320
    
    sketches = [convert(sketch, 'zh-cn') for sketch in examples['sketch']]
    texts = [convert(text, 'zh-cn') for text in examples['text']]
    
    model_inputs = tokenizer(sketches,
                             max_length=source_max_length,
                             padding=True,
                             truncation=True,
                             return_tensors='pt')
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(texts,
                           max_length=target_max_length,
                           padding=True, 
                           truncation=True,
                           return_tensors='pt')
    model_inputs['labels'] = labels['input_ids']
    return model_inputs


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # print("predictions: {}, labels: {}".format(predictions, labels))
    predictions = np.argmax(predictions.detach.cpu().numpy(), axis=1)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip()))
                      for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) 
                      for label in decoded_labels]
    
    # Compute ROUGE scores
    result = metric.compute(predictions=decoded_preds, references=decoded_labels,
                            use_stemmer=True)

    # Extract ROUGE f1 scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length to metrics
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id)
                      for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    # print("result: {}".format(result))
    
    return {k: round(v, 4) for k, v in result.items()}


# load model and tokenizer
pretrained_model = "Maciel/T5_Mask_Completion"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model)
data_collator = DataCollatorForSeq2Seq(tokenizer)
pretrained_model_name = pretrained_model.split("/")[-1]
logger.info("pretrained_model_name: {}, load model and tokenizer done!".format(pretrained_model_name))

# load dataset
dataset_path = '../saved_datasets/chinese_4kw_sketches' 
dataset_name = dataset_path.split('/')[-1]
dataset_with_sketch = load_from_disk(dataset_path)

train_dataset = dataset_with_sketch['train']
train_dataset = train_dataset.filter(lambda example: len(example['text']) < 100 and len(example['sketch']) < 100)
eval_dataset = dataset_with_sketch['test']
eval_dataset = eval_dataset.filter(lambda example: len(example['text']) < 100 and len(example['sketch']) < 100)
logger.info("train dataset num: {}, eval dataset num: {}".format(len(train_dataset), len(eval_dataset)))

# mapping train and eval dataset with tokenizer
train_sketch_dataset = train_dataset.map(
                            preprocess_function,
                            batched=True,
                            remove_columns=train_dataset.column_names,
                            batch_size=50,
                            num_proc=20)
eval_sketch_dataset = eval_dataset.map(
                            preprocess_function, 
                            batched=True, 
                            remove_columns=eval_dataset.column_names,
                            batch_size=50,
                            num_proc=20)
logger.info("tokenizer train and eval dataset done!")

metric = load_metric("rouge")
logger.info("load rouge metric done!")

saved_dir =  "../models/{}_4kw_sketch".format(pretrained_model_name)
train_batch_size = 32
eval_batch_size = 32
args = TrainingArguments(
    output_dir=saved_dir, #The output directory
    overwrite_output_dir=True, #overwrite the content of the output directory
    num_train_epochs=3, # number of training epochs
    per_device_train_batch_size=train_batch_size, # batch size for training
    per_device_eval_batch_size=eval_batch_size,  # batch size for evaluation
    eval_steps=5000, # Number of update steps between two evaluations.
    save_steps=5000, # after # steps model is saved 
    #logging_steps=5000,
    warmup_steps=5000,# number of warmup steps for learning rate scheduler
    prediction_loss_only=True,
    save_total_limit=3,
    load_best_model_at_end=True,
    save_strategy="steps",
    evaluation_strategy="steps",
    gradient_accumulation_steps=8,
    learning_rate=6e-5,
    weight_decay=0.01,
    fp16=True,
    report_to="tensorboard"
)

# args = Seq2SeqTrainingArguments(
#     output_dir=saved_dir,
#     evaluation_strategy="steps",
#     eval_steps = 10000,      
#     save_strategy = 'epoch',
#     save_total_limit = 3,
#     fp16 = True,
#     learning_rate=5.6e-5,
#     per_device_train_batch_size=train_batch_size,
#     per_device_eval_batch_size=eval_batch_size,
#     weight_decay=0.01,
#     num_train_epochs=5,
#     predict_with_generate=True,
#     logging_steps=len(train_dataset) // train_batch_size
# )

trainer = Trainer(
    model,
    args=args,
    train_dataset=train_sketch_dataset,
    eval_dataset=eval_sketch_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    # compute_metrics=compute_metrics
)        

logger.info("start train model...")
trainer.train()
trainer.save_model()