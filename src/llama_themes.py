import torch
import pickle
import transformers
from transformers import AutoTokenizer

import pandas as pd
from tqdm import tqdm
from collections import Counter


def themes_info(texts):
    thematics = []
    for index, row in texts.iterrows():
        thematic = row['thematic'].lower()
        thematics.append(thematic)
    print('Number of unique themes =', len(set(thematics)))
    print()
    print('The largest areas of knowledge in data:')
    for unit in Counter(thematics).most_common(5):
        print(f'Area = {unit[0]}, Number of texts = {unit[1]}')
    return thematics


if __name__ == "__main__":
    file_path='eng_texts_terms_science(1).jsonl' # for eng
    en_df = pd.read_json(path_or_buf=file_path, lines=True)
    print('Language: English')
    en_thematics = themes_info(en_df)

    # Output:
    # Language: English
    # Number of unique themes = 1792

    # The largest areas of knowledge in data:
    # Area = medicine, Number of texts = 68
    # Area = computer science, Number of texts = 39
    # Area = chemistry, Number of texts = 27
    # Area = materials science, Number of texts = 20
    # Area = psychology, Number of texts = 20

    file_path='rus_texts_terms_science(1).jsonl' # for ru
    ru_df = pd.read_json(path_or_buf=file_path, lines=True)
    print('Language: Russian')
    ru_thematics = themes_info(ru_df)

    # Output:
    # Language: Russian
    # Number of unique themes = 704

    # The largest areas of knowledge in data:
    # Area = humanities, Number of texts = 452
    # Area = physics, Number of texts = 283
    # Area = political science, Number of texts = 115
    # Area = mathematics, Number of texts = 80
    # Area = medicine, Number of texts = 62

    new_en_thematics = []
    new_ru_thematics = []

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    for theme in tqdm(en_thematics):
        messages = [
            {"role": "system", "content": "You are a chatbot who helps people."},
            {"role": "user", "content": f'You will see the theme of some scientific paper. It can be either a common area of knowledge, like "psychology" or "computer science" or some strange area, like "internal medicine". You should either return a name as it is, if the theme is a valid basic science direction, or summarise the narrow theme into a broader scientific direction. For example, for "computer science" you should return "computer science", but for "internal medicine" you should return just "medicine". Write only the result theme. My theme: "{theme}", your answer (the result theme):  \n'},
        ]

        prompt = pipeline.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
        )

        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = pipeline(
            prompt,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        new_en_thematics.append(outputs[0]["generated_text"][len(prompt):])

    with open('llama3_en_thematics.ob', 'wb') as fp:
        pickle.dump(new_en_thematics, fp)

    for theme in tqdm(ru_thematics):
        messages = [
            {"role": "system", "content": "You are a chatbot who helps people."},
            {"role": "user", "content": f'You will see the theme of some scientific paper. It can be either a common area of knowledge, like "psychology" or "computer science" or some strange area, like "internal medicine". You should either return a name as it is, if the theme is a valid basic science direction, or summarise the narrow theme into a broader scientific direction. For example, for "computer science" you should return "computer science", but for "internal medicine" you should return just "medicine". Write only the result theme. My theme: "{theme}", your answer (the result theme):  \n'},
        ]

        prompt = pipeline.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
        )

        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = pipeline(
            prompt,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        new_ru_thematics.append(outputs[0]["generated_text"][len(prompt):])

    with open('llama3_ru_thematics.ob', 'wb') as fp:
        pickle.dump(new_ru_thematics, fp)

