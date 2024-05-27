import os
import sys

import torch
import pandas as pd
from ast import literal_eval

import numpy as np
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    set_seed, 
)

from config import max_seq_length

from pathlib import Path
from tqdm import tqdm

from datasets import Dataset


en_df = pd.read_csv('en_df.csv', converters={'text': literal_eval, 'terms': literal_eval})
ru_df = pd.read_csv('ru_df.csv', converters={'text': literal_eval, 'terms': literal_eval})

print(ru_df.head())

for index, row in tqdm(ru_df.iterrows()):
    assert len(row['text']) == len(row['terms'])

for index, row in tqdm(en_df.iterrows()):
    assert len(row['text']) == len(row['terms'])

set_seed(41)
cache_dir="./cache"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

label_list=['not_term', 'term']
label_to_id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label_to_id.items()}

en_model_names = [
'google-bert/bert-base-uncased',
'google-bert/bert-base-multilingual-uncased',
'openai-community/gpt2',
]
ru_model_names = [
'DeepPavlov/rubert-base-cased',
'google-bert/bert-base-multilingual-uncased',
'ai-forever/rugpt3small_based_on_gpt2']

def get_model_tokenizer(model_name):
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name,
        config=config,
        cache_dir=cache_dir,
        torch_dtype=config.torch_dtype).to(device)
    
    if config.model_type in {"gpt2", "roberta"}:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=True, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=True)
        
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token  
            print(f'Set pad_token to be equal to eos_token: {tokenizer.pad_token}')
        else:
            raise ValueError("The tokenizer does not have an eos_token set.")
    return model, tokenizer

def tokenize_and_align_labels(examples, model_name, tokenizer, label_all_tokens=False):
    tokenized_inputs = tokenizer(
        examples['text'],
        truncation=True,
        max_length=max_seq_length[model_name],
        is_split_into_words=True,
    )
    labels = []
    for i, label in enumerate(examples['terms']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                if label_all_tokens:
                    label_ids.append(label_to_id[label[word_idx]])
                else:
                    label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

ru_dataset = Dataset.from_pandas(ru_df)
en_dataset = Dataset.from_pandas(en_df)

from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
        def __init__(self, input_ids, attention_masks, labels):
            self.input_ids = input_ids
            self.attention_masks = attention_masks
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                'input_ids': self.input_ids[idx],
                'attention_mask': self.attention_masks[idx],
                'labels': self.labels[idx]
            }

def safe_indexing(y): # converts tensor to int if necessary
    return y.item() if isinstance(y, torch.Tensor) else y 

def extract_representations(model_name, model, data, language='en'):

    input_ids = [torch.tensor(d['input_ids']) for d in data]
    attention_masks = [torch.tensor(d['attention_mask']) for d in data]
    labels = [d['labels'] for d in data]  
    
    custom_dataset = CustomDataset(input_ids, attention_masks, labels)
    dataloader = DataLoader(custom_dataset, batch_size=1, shuffle=False)

    first_token = []
    mean_embeddings = []
    label_list = []

    model.eval()
    with torch.no_grad():  
        for batch in tqdm(dataloader, desc="Processing batches"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch['labels'] = [safe_indexing(y) for y in batch['labels']]
            labels = torch.tensor(batch['labels']).to(device)
            
            if input_ids is None or input_ids.dtype not in (torch.long, torch.int):
                print("Skipping batch because input_ids is None or not of type Long or Int.")
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                hidden_states = outputs[0]

                ### normal average
                num_tokens = torch.sum(attention_mask, dim=-1).unsqueeze(-1)
                mean_embedding = torch.sum(hidden_states * attention_mask.unsqueeze(-1), dim=1)
                mean_embedding = mean_embedding / num_tokens
                assert torch.allclose(torch.mean(hidden_states, 1), mean_embedding, rtol=0.000001)
                mean_embeddings.append(mean_embedding.cpu())
                ### first token
                hidden_states = hidden_states.squeeze(0)
                # labels not equal to -100
                valid_indices = labels != -100
                # apply the mask to the hidden_states and labels
                filtered_hidden_states = hidden_states[valid_indices]
                filtered_labels = labels[valid_indices]
                first_token.append(filtered_hidden_states.cpu())
                label_list.extend(filtered_labels.cpu())
                
    final_first_embeddings = torch.cat(first_token, dim=0)
    final_mean_embeddings = torch.cat(mean_embeddings, dim=0)
    print(final_first_embeddings.shape)
    print(final_mean_embeddings.shape)
    assert final_mean_embeddings.shape[0] == 2500
    model_name = model_name.replace('/', '_')  
    current_directory = Path(os.getcwd())
    data_repr_dir = current_directory / 'token_repr' / f'{language}_data'
    model_name_dir = data_repr_dir / model_name
    model_name_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_name_dir / f"{language}_data.pt"
    torch.save({'features': final_first_embeddings, 'mean_embeddings': final_mean_embeddings, 'labels': label_list, 'label_to_id': label_to_id}, save_path)

for model_name in ru_model_names:
    model, tokenizer = get_model_tokenizer(model_name)
    ru_train_dataset = ru_dataset.map(lambda examples: tokenize_and_align_labels(examples, model_name, tokenizer), batched=True)
    for i in range(len(ru_train_dataset)):
        if len(ru_train_dataset[i]['input_ids']) != max_seq_length[model_name]:
            assert sum(torch.tensor(ru_train_dataset[i]['labels'])!=-100).item() == len(ru_train_dataset[i]['terms'])
    print(sum(ru_df['text_length']))
    extract_representations(model_name, model, ru_train_dataset, language='ru')

for model_name in en_model_names:
    model, tokenizer = get_model_tokenizer(model_name)
    en_train_dataset = en_dataset.map(lambda examples: tokenize_and_align_labels(examples, model_name, tokenizer), batched=True)
    for i in range(len(en_train_dataset)):
        if len(en_train_dataset[i]['input_ids']) != max_seq_length[model_name]:
            assert sum(torch.tensor(en_train_dataset[i]['labels'])!=-100).item() == len(en_train_dataset[i]['terms'])
    print(sum(en_df['text_length']))
    extract_representations(model_name, model, en_train_dataset, language='en')