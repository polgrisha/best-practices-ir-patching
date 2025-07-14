import os
from fire import Fire 
import ir_datasets as irds
import pandas as pd
import pyterrier as pt
from pyterrier_dr import HgfBiEncoder, ElectraScorer, SimFn
if not pt.started():
    pt.init()

DL19 = r"msmarco-passage/trec-dl-2019/judged"
DL20 = r"msmarco-passage/trec-dl-2020/judged"
MSMARCO = r"msmarco-passage/train/triples-small"
MSMARCO_TERRIER = r"msmarco_passage"

def load_bi(model_name_or_path : str, batch_size : int = 256, sim_fn: str = 'dot'):
    # Hacky wat to get the scorer TODO: Fix this
    sim_fn = {'cos': SimFn.cos, 'dot': SimFn.dot}[sim_fn]
    return HgfBiEncoder.from_pretrained(model_name_or_path, batch_size=batch_size).scorer(batch_size=batch_size, sim_fn=sim_fn)

def load_cross(model_name_or_path : str, batch_size : int = 256):
    return ElectraScorer(model_name_or_path, batch_size=batch_size, verbose=True)

def topk(model_name_or_path : str, model_type : str, in_file : str, out_path : str, k : int = 1000, batch_size : int = 256, perturbation_type : str = 'TFC1', sim_fn: str = 'dot', max_rel : int = 0):
    formatted_model_mame = model_name_or_path.replace("/", "-")
    output_all_file = f"{out_path}/{formatted_model_mame}_{model_type}_{perturbation_type}_all_msmarco.tsv"
    if os.path.exists(output_all_file):
        full_deltas = pd.read_csv(output_all_file, sep='\t')
    else:
        if model_type == "bi":
            model = load_bi(model_name_or_path, batch_size, sim_fn)
        elif model_type == "cross":
            model = load_cross(model_name_or_path, batch_size)
        else:
            raise ValueError("model_type must be either 'bi' or 'cross'")
        
        all_data = pd.read_csv(in_file, sep='\t')

        text_lookup = all_data.set_index(['qid', 'docno', 'perturbed']).text.to_dict()
        query_text_lookup = all_data.set_index(['qid'])['query'].to_dict()

        scored_data = model.transform(all_data)

        all_deltas = []
        for rel_grade in range(max_rel + 1):
            rel_data = scored_data[scored_data.relevance == rel_grade]
            # original_scores = rel_data[~rel_data.perturbed].set_index(['qid', 'docno'])['score']
            # perturbed_scores = rel_data[rel_data.perturbed].set_index(['qid', 'docno'])['score']
            
            # score_deltas = (perturbed_scores - original_scores).reset_index()
            # score_deltas.columns = ['qid', 'docno', 'score_delta']
            
            original_data = rel_data[~rel_data.perturbed].rename(columns={'score': 'original_score'})
            perturbed_data = rel_data[rel_data.perturbed].rename(columns={'score': 'perturbed_score'})
            score_deltas = perturbed_data.merge(original_data[['qid', 'docno', 'original_score']], on=['qid', 'docno'])
            score_deltas['score_delta'] = score_deltas.perturbed_score - score_deltas.original_score
            
            score_deltas['text'] = score_deltas.apply(lambda x : text_lookup[(x.qid, x.docno, False)], axis=1)
            score_deltas['perturbed_text'] = score_deltas.apply(lambda x : text_lookup[(x.qid, x.docno, True)], axis=1)
            score_deltas['query'] = score_deltas.apply(lambda x : query_text_lookup[x.qid], axis=1)
            score_deltas['relevance'] = rel_grade
            
            all_deltas.append(score_deltas)
        
        # Combine all deltas
        full_deltas = pd.concat(all_deltas, ignore_index=True)
    
    # Get top-k from each relevance grade
    topk_results = []
    for rel_grade in range(max_rel + 1):
        rel_deltas = full_deltas[full_deltas.relevance == rel_grade]
        top_k = rel_deltas.nlargest(k // (max_rel+1), 'score_delta')
        topk_results.append(top_k)
    
    output_k_file = f"{out_path}/{formatted_model_mame}_{model_type}_{perturbation_type}_topk_{k}_msmarco.tsv"

    topk_df = pd.concat(topk_results)
    topk_df.to_csv(output_k_file, sep='\t', index=False)
    full_deltas.to_csv(output_all_file, sep='\t', index=False)

    print(f"Top-k results saved to {output_k_file}")
    print(f"Full results saved to {output_all_file}")

    return 0

if __name__ == "__main__":
    Fire(topk)