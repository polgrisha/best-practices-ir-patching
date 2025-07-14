from fire import Fire 
import ir_datasets as irds
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import mechir
from mechir import Cat, Dot
from mechir.data import CatDataCollator, DotDataCollator, MechDataset
import pyterrier as pt
if not pt.started():
    pt.init()

mechir.config('ignore-official', True)

def load_bi(model_name_or_path : str, sim_fn: str = 'dot'):
    return Dot(model_name_or_path, sim_fn=sim_fn), DotDataCollator

def load_cross(model_name_or_path : str):
    return Cat(model_name_or_path), CatDataCollator

def process_frame(frame):

    output = {
        'qid': [],
        'query': [],
        'docno': [],
        'text': [],
        'perturbed': [],
    }

    for row in frame.itertuples():
        output['qid'].append(row.qid)
        output['query'].append(row.query)
        output['docno'].append(row.docno)
        output['text'].append(row.text)
        output['perturbed'].append(row.perturbed_text)
    
    return pd.DataFrame(output)

def patch(model_name_or_path : str, 
          model_type : str, 
          in_file : str, 
          out_path : str, 
          batch_size : int = 256, 
          perturbation_type : str = 'TFC1', 
          sim_fn: str = 'dot', 
          k : int = 1000, 
          save_indivudual_samples: bool = False,
          patch_type: str = "head_all"):
    if model_type == "bi":
        model, collator = load_bi(model_name_or_path, sim_fn=sim_fn)
    elif model_type == "cross":
        model, collator = load_cross(model_name_or_path)
    else:
        raise ValueError("model_type must be either 'bi' or 'cross'")
        
    all_data = pd.read_csv(in_file, sep='\t')
    processed_frame = process_frame(all_data)

    dataset = MechDataset(processed_frame, pre_perturbed=True)
    collator = collator(model.tokenizer, pre_perturbed=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collator)

    patching_head_outputs = []
    all_original_scores = []
    all_perturbed_scores = []
    i = 0
    for batch in tqdm(dataloader):
        # Get the queries, documents, and perturbed documents from the batch

        if model_type == "bi":
            queries = batch["queries"]
            documents = batch["documents"]
            perturbed_documents = batch["perturbed_documents"]

            patch_head_out, original_scores, perturbed_scores = model(queries, documents, perturbed_documents, patch_type=patch_type)
        else:
            sequences = batch["sequences"]
            perturbed_sequences = batch["perturbed_sequences"]

            patch_head_out, original_scores, perturbed_scores = model(sequences, perturbed_sequences, patch_type=patch_type)

        patching_head_outputs.append(patch_head_out.cpu().detach())
        all_original_scores.append(original_scores.cpu().detach())
        all_perturbed_scores.append(perturbed_scores.cpu().detach())

    # output = torch.mean(torch.stack(patching_head_outputs), axis=0)
    output = torch.cat(patching_head_outputs, dim=0)
    all_original_scores = torch.cat(all_original_scores, dim=0)
    all_perturbed_scores = torch.cat(all_perturbed_scores, dim=0)
    
    # convert to numpy and dump
    formatted_model_name = model_name_or_path.replace("/", "-")
    output_file = f"{out_path}/{formatted_model_name}_{model_type}_{perturbation_type}_{k}_batch_size_{batch_size}_patch_{patch_type}_msmarco.npy"
    np.save(output_file, output)
    np.save(output_file.replace(f"patch_{patch_type}", f"patch_{patch_type}_original_scores"), all_original_scores)
    np.save(output_file.replace(f"patch_{patch_type}", f"patch_{patch_type}_perturbed_scores"), all_perturbed_scores)
    return 0

if __name__ == "__main__":
    Fire(patch)