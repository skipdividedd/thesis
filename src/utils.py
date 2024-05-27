import torch
import pandas as pd
from ast import literal_eval
from collections import Counter

en_df = pd.read_csv('en_df.csv', converters={'text': literal_eval, 'terms': literal_eval})
ru_df = pd.read_csv('ru_df.csv', converters={'text': literal_eval, 'terms': literal_eval})

en_themes = en_df['theme'].tolist()
ru_themes = ru_df['theme'].tolist()

path_config = {
'en_multi_bert': 'token_repr/en_data/google-bert_bert-base-multilingual-uncased/en_data.pt',
'en_bert': 'token_repr/en_data/google-bert_bert-base-uncased/en_data.pt',
'en_gpt': 'token_repr/en_data/openai-community_gpt2/en_data.pt', 
'ru_multi_bert': 'token_repr/ru_data/google-bert_bert-base-multilingual-uncased/ru_data.pt',
'ru_bert': 'token_repr/ru_data/DeepPavlov_rubert-base-cased/ru_data.pt',
'ru_gpt': 'token_repr/ru_data/ai-forever_rugpt3small_based_on_gpt2/ru_data.pt', 
}

def safe_indexing(y): # converts tensor to int if necessary
    return y.item() if isinstance(y, torch.Tensor) else y 

def theme_labelling(counts, themes):
    new_labels = []
    for i, word in enumerate(themes):
        new_labels.extend([word] * counts[i])
    return new_labels

def load_data(train, data_type= 'sentence', language='en'):
    if data_type == 'sentence':
        X = train['mean_embeddings']
        if language == 'en':
            y = en_themes
        else:
            y = ru_themes
        assert len(X) == len(y)
        theme_counts = Counter(y)
        most_common_themes = [theme for theme, count in theme_counts.most_common(10)]
        most_common_themes_set = set(most_common_themes)
        filtered_X = [vector for vector, theme in zip(X, y) if theme in most_common_themes_set]
        filtered_y = [theme for theme in y if theme in most_common_themes_set]
        assert len(filtered_X) == len(filtered_y)
        return filtered_X, filtered_y
    else:
        X = train['features']
        original_y = [safe_indexing(y) for y in train['labels']]
        filtered_X = [x for x, label in zip(X, original_y) if label == 1]
        filtered_y = [label for label in original_y if label == 1]
        assert len(filtered_X) == len(filtered_y)
        if language == 'en':
            themes = theme_labelling(en_df['text_length'].tolist(), en_themes)
        else:
            themes = theme_labelling(ru_df['text_length'].tolist(), ru_themes)
        theme_counts = Counter(themes)
        most_common_themes = [theme for theme, count in theme_counts.most_common(10)]
        most_common_themes_set = set(most_common_themes)
        filtered_themes = [theme for theme, label in zip(themes, original_y) if label == 1]
        assert len(filtered_themes) == len(filtered_X) == len(filtered_y)
        finally_filtered_X = [vector for vector, theme in zip(filtered_X, filtered_themes) if theme in most_common_themes_set]
        finally_filtered_y = [theme for theme in filtered_themes if theme in most_common_themes_set]
        assert len(finally_filtered_X) == len(finally_filtered_y)
        return finally_filtered_X, finally_filtered_y