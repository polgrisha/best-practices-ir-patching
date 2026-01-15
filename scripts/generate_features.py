import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import tokenize_and_stem, get_mean_tfidf, get_std_tfidf, get_english_probability, get_readability_scores, calculate_perplexity
import ir_datasets as irds
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import AutoTokenizer, AutoModelForCausalLM
from functools import partial
from pyserini.index.lucene import LuceneIndexReader as IndexReader
from tqdm.auto import tqdm
tqdm.pandas()


QUERY_DOC_PAIRS_PATH = "./data/diagnostic_dataset/TFC1-data.tsv.gz"
SAVE_PATH_QUERIES = "./data/diagnostic_dataset/queries_with_features.tsv"
SAVE_PATH_DOCUMENTS = "./data/diagnostic_dataset/documents_with_features.tsv"


def compute_document_features(
    dataframe: pd.DataFrame,
    reader: IndexReader,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    suffix: str = ''
) -> pd.DataFrame:
    """
    Compute several textual features for the given text column suffix.
    """
    print(f"Computing TF-IDF statistics{suffix}...")
    dataframe['tfidf_mean' + suffix] = dataframe['text_tokenized' + suffix].progress_apply(lambda x: get_mean_tfidf(x, reader))
    dataframe['tfidf_std' + suffix] = dataframe['text_tokenized' + suffix].progress_apply(lambda x: get_std_tfidf(x, reader))

    print(f"Computing length and language metrics{suffix}...")
    dataframe['doc_length' + suffix] = dataframe['text_tokenized' + suffix].apply(lambda x: len(x.split()))
    dataframe['english_probability' + suffix] = dataframe['text_tokenized' + suffix].progress_apply(get_english_probability)
    dataframe['num_sentences' + suffix] = dataframe['text' + suffix].progress_apply(lambda x: len(sent_tokenize(x)))

    print(f"Computing readability scores{suffix}...")
    readability_scores = dataframe['text' + suffix].progress_apply(get_readability_scores)
    readability_df = pd.DataFrame(readability_scores.tolist(), index=readability_scores.index)
    dataframe = pd.concat([dataframe, readability_df], axis=1)

    print(f"Computing decoder perplexity{suffix}...")
    # dataframe['decoder_perplexity' + suffix] = calculate_perplexity(dataframe, model, tokenizer)

    return dataframe


if __name__ == "__main__":    
    # Load existing query-document pairs
    query_doc_pairs = pd.read_csv(QUERY_DOC_PAIRS_PATH, sep='\t')
    documents = query_doc_pairs[query_doc_pairs['perturbed'] == False][['docno', 'text']].drop_duplicates()
    queries = query_doc_pairs[query_doc_pairs['perturbed'] == False][['qid', 'query']].drop_duplicates()
    queries.columns = ['query_id', 'text_query']
    
    # Initialize stemmer
    stemmer = PorterStemmer()
    
    # Intialize index reader for TF-IDF calculations
    reader = IndexReader.from_prebuilt_index('msmarco-v1-passage-slim')
    
    # Initialize GPT-2 model and tokenizer for perplexity calculations
    model_name = 'gpt2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if model_name == 'gpt2':
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    model.eval()
    
    # Tokenize and stem the documents
    print('Tokenizing documents...')
    documents['text_tokenized'] = documents['text'].progress_apply(tokenize_and_stem)
    documents['text_stemmed'] = documents['text'].progress_apply(partial(tokenize_and_stem, stemmer=stemmer))
    
    # Compute document features
    print('Computing document features...')
    documents = compute_document_features(documents, reader, model, tokenizer)
    
    # Tokenize and stem queries
    print('Tokenizing queries...')
    queries['text_tokenized_query'] = queries['text_query'].progress_apply(tokenize_and_stem)
    queries['text_stemmed_query'] = queries['text_query'].progress_apply(partial(tokenize_and_stem, stemmer=stemmer))
    
    # Compute query features
    print('Computing query features...')
    queries = compute_document_features(queries, reader, model, tokenizer, suffix='_query')
    
    # Save queries and documents with features
    print('Saving queries and documents with features...')
    queries.to_csv(SAVE_PATH_QUERIES, sep='\t', index=False)
    documents.to_csv(SAVE_PATH_DOCUMENTS, sep='\t', index=False)
    
    # # Merge features back to the original dataframe
    # print('Merging features back to the original dataframe...')
    # query_doc_pairs = query_doc_pairs.merge(
    #     documents,
    #     left_on='docno',
    #     right_on='docno',
    #     how='left'
    # )
    # query_doc_pairs = query_doc_pairs.merge(
    #     queries,
    #     left_on='qid',
    #     right_on='qid',
    #     how='left',
    #     suffixes=('', '_query')
    # )
    
    # print(query_doc_pairs)
    
    