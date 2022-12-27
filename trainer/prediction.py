import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import Text2TextGenerationPipeline
import os
import readline
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def generate_sentences(generator, sentence):
    generations = generator(sentence,
                            max_length=100,
                            do_sample=True,
                            num_beams=3)
    generated_text = generations[0]['generated_text'].replace(" ", "")
    confidence = -1
    return generated_text, confidence


def generate_sentences2(model, tokenizer, sentence, device):
    input_encodings = tokenizer(sentence, 
                                max_length=max_input_length, 
                                truncation=True, 
                                return_tensors="pt")
    if "token_type_ids" in input_encodings.keys():
        input_encodings.pop("token_type_ids")
    output = model.generate(**input_encodings, 
                            num_beams=5,
                            no_repeat_ngram_size=5,
                            do_sample=True, 
                            early_stopping=True,
                            min_length=10,
                            max_length=128,
                            return_dict_in_generate=True,
                            output_scores=True)
    decoded_output = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)[0]
    decoded_scores = output.sequences_scores
    confidence = torch.exp(decoded_scores).item()
    # generation = nltk.sent_tokenize(decoded_output.strip())[0]
    generated_text = decoded_output.strip().split("</s>")[0]
    return generated_text, confidence
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# saved_model_path = "../models/mengzi-t5-base/checkpoint-185000"
saved_model_path = "Maciel/T5_Mask_Completion"
tokenizer = AutoTokenizer.from_pretrained(saved_model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(saved_model_path)
generator = Text2TextGenerationPipeline(model, tokenizer, device)
print("load model and tokenizer done!")
    
max_input_length = 128
while True:
    print("input your sentence: ...")
    sentence = input()
    print("sentence: {}\n".format(sentence))
    generated_text1, confidence = generate_sentences(generator, sentence)
    print("generation1: {}\n".format(generated_text1))
    generated_text2, confidence = generate_sentences2(model, tokenizer, sentence, device)
    print("generation2: {}\nconfidence: {}\n".format(generated_text2, confidence))
