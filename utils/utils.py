import torch
import numpy as np
import pandas as pd
from collections import Counter
from langdetect import detect_langs
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from language_tool_python import LanguageTool
import math
import re
from textstat import textstat
from tqdm.auto import tqdm
from pyserini.index.lucene import LuceneIndexReader as IndexReader


tool = LanguageTool('en-US')


def tokenize_and_stem(text, stemmer=None):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    if stemmer is not None:
        processed_tokens = [stemmer.stem(word) for word in tokens]
    else:
        processed_tokens = tokens
    return ' '.join(processed_tokens)


def get_english_probability(text):
    eng_prob = 0.0
    try:
        lang_probs = detect_langs(text)
        for prob in lang_probs:
            if prob.lang == 'en':
                eng_prob = prob.prob
    except:
        pass
    return eng_prob


def get_grammar_score(text):
    matches = tool.check(text)
    word_count = len(text.split()) # this is not a good way to count words TODO improve
    error_count = len(matches)
    error_ratio = error_count / word_count
    return error_ratio


def get_tfidf(text, reader):
    terms = text.split()
    tf = Counter(terms)
    num_docs = reader.stats()['documents']
    
    tfidf_scores = []
    for term, term_count in tf.items():
        try:
            df = reader.get_term_counts(term)[0]
        except:
            df = 0
        if df > 0:
            tf = term_count / len(terms)
            idf = math.log(num_docs / df)
            tfidf_scores.append(tf * idf)
    return tfidf_scores


def get_ifd_injected_term(term, reader):
    try:
        df = reader.get_term_counts(term)[0]
        num_docs = reader.stats()['documents']
        idf = math.log(num_docs / df)
        return idf
    except:
        return 0


def get_tf_injected_term(text, term):
    terms = text.split()
    tf = Counter(terms)
    if term in tf:
        return tf[term]
    else:
        return 0


def get_tfidf_injected_term(text, term, reader):
    tfidf_scores = get_tfidf(text, reader)
    if term in tfidf_scores:
        return tfidf_scores[term]
    else:
        return 0


def get_mean_tfidf(text, reader):
    tfidf = get_tfidf(text, reader)
    return np.mean(tfidf) if tfidf else 0.0


def get_std_tfidf(text, reader):
    tfidf = get_tfidf(text, reader)
    return np.std(tfidf) if tfidf else 0.0


def get_readability_scores(text, suffix=''):
    text = text.lower()
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:\'\"\-]', '', text)
    
    return {
        'flesch_reading_ease' + suffix: textstat.flesch_reading_ease(text),
        'flesch_kincaid_grade' + suffix: textstat.flesch_kincaid_grade(text),
        'gunning_fog' + suffix: textstat.gunning_fog(text),
        'smog_index' + suffix: textstat.smog_index(text),
        'coleman_liau_index' + suffix: textstat.coleman_liau_index(text),
        'automated_readability_index' + suffix: textstat.automated_readability_index(text),
        'dale_chall_readability_score' + suffix: textstat.dale_chall_readability_score(text)
    }
    
    
def calculate_perplexity_batch(texts, model, tokenizer, max_length=512):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, 
                       truncation=True, max_length=max_length)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)
    with torch.no_grad():
        outputs = model(input_ids = input_ids, 
                        attention_mask = attention_mask)
        
        target_ids = input_ids[:, 1:]
        logits = outputs.logits[:, :-1, :]
        attention_mask = attention_mask[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)

        batch_size, seq_length = target_ids.shape
        batch_indices = torch.arange(batch_size).unsqueeze(1).to(model.device)
        seq_indices = torch.arange(seq_length).unsqueeze(0).to(model.device)
        token_log_probs = log_probs[batch_indices, seq_indices, target_ids]
        
        cross_entropy = -torch.sum(token_log_probs * attention_mask, dim=1) / attention_mask.sum(dim=1)
        perplexity = torch.exp2(cross_entropy)
            
    return perplexity.cpu().numpy().tolist()


def calculate_perplexity(df, model, tokenizer, text_column='text', batch_size=32):    
    entropies = []
    for i in tqdm(range(0, len(df), batch_size)):
        batch_texts = df[text_column].iloc[i:i+batch_size].tolist()
        batch_entropies = calculate_perplexity_batch(batch_texts, model, tokenizer)
        entropies.extend(batch_entropies)

    return pd.Series(entropies, index=df.index, name='model_perplexity')


def compute_perturbation_featues(data, tokenizer):
    # Check if required columns exist
    if 'injected_term' not in data.columns:
        raise ValueError("Error: The required column 'injected_term' is missing from the DataFrame.")
    
    # Intialize index reader for TF-IDF calculations
    reader = IndexReader.from_prebuilt_index('msmarco-v1-passage-slim')

    # Compute perturbation features
    data['idf_injected_term'] = data['injected_term'].apply(lambda x: get_ifd_injected_term(x, reader))
    data['tf_injected_term'] = data.apply(lambda x: get_tf_injected_term(x['text_tokenized'], x['injected_term']), axis=1)
    data['tfidf_injected_term'] = data.apply(lambda x: get_tfidf_injected_term(x['text_tokenized'], x['injected_term'], reader), axis=1)
    data['injected_term_token_len'] = data['injected_term'].apply(lambda x: len(tokenizer(x, add_special_tokens=False)['input_ids']))
    
    return data