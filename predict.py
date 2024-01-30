import os
import argparse
import json
from collections import Counter, defaultdict

import tqdm
import torch
from fairseq.checkpoint_utils import load_model_ensemble_and_task

import ail2abel # noqa


def preprocess_samples(samples, dictionary_dict):
    preprocessed_samples = []
    for sample in samples:
        preprocessed_sample = {}
        for key, value in sample.items():
            if key == "ail_variables":
                continue
            dictionary = dictionary_dict[key]
            indices = [dictionary.index(token) for token in value.split(" ")]
            preprocessed_sample[key] = torch.LongTensor(indices)
        preprocessed_samples.append(preprocessed_sample)
    return preprocessed_samples


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--json_dir", type=str)
    # parser.add_argument("--out_dir", type=str)
    # parser.add_argument("--checkpoint", type=str)
    # args = parser.parse_args()

    json_dir = "./outputs"
    out_dir = "./predicteds"
    checkpoint = "checkpoints/ailabel_x64_O2_270w_100pp/checkpoint_best.pt"
    

    os.makedirs(out_dir, exist_ok=True)
    models, saved_cfg, task = load_model_ensemble_and_task([checkpoint])
    model = models[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    for binary_dir in tqdm.tqdm(os.listdir(json_dir)):
        for func_json_file in os.listdir(os.path.join(json_dir, binary_dir)):
            if not func_json_file.endswith(".json"):
                continue

            func_json_path = os.path.join(json_dir, binary_dir, func_json_file)
            predicted_path = os.path.join(out_dir, binary_dir, func_json_file)
            os.makedirs(os.path.dirname(predicted_path), exist_ok=True)

            with open(func_json_path, "r") as f:
                samples = json.load(f)
            preprocessed_samples = preprocess_samples(samples, task.dictionary_dict)
            
            predicted_samples = []

            mask_idx = task.source_dictionary["ail_token"].index("<mask>")
            for sample, preprocessed_sample in zip(samples, preprocessed_samples):
                preprocessed_sample = {
                    k: v.unsqueeze(0).to(device) for k, v in preprocessed_sample.items()
                }
                masked_code = preprocessed_sample["ail_token"].eq(mask_idx)

                try:
                    output, _ = model(
                        src_tokens=preprocessed_sample,
                        masked_code=masked_code,
                        classification_head_name="maskvar"
                    )
                except Exception as e:
                    print(f"Inference error when processing {func_json_path}: {e}")
                    continue

                predicteds = torch.argmax(output, dim=-1).reshape(-1, 4)
                predicted_tokens = task.source_dictionary["ail_token_label"].string(predicteds)

                vids = []
                for idx, mask in enumerate(masked_code[0]):
                    if idx == 0:
                        continue
                    if mask and not masked_code[0][idx-1]:
                        vids.append(sample["ail_token"].split()[idx - 1])
                
                assert len(vids) == len(predicteds)
                id2varlabels = defaultdict(list)
                for vid, predicted in zip(vids, predicteds):
                    predicted_tokens = task.source_dictionary["ail_token_label"].string(predicted)
                    predicted_labels = [token for token in predicted_tokens.split() if token != "<vpad>"]
                    id2varlabels[vid].append(predicted_labels)
                
                # vote for each vid and pick the most common one
                for vid, varlabels in id2varlabels.items():
                    vote_conuter = Counter()
                    for var_label in varlabels:
                        vote_conuter.update(var_label)
                    id2varlabels[vid] = vote_conuter
                
                predicted_sample = sample.copy()
                for vid in predicted_sample["ail_variables"]:
                    predicted_sample["ail_variables"][vid]["predicted_labels"] = id2varlabels[vid]
            
                predicted_samples.append(predicted_sample)

            with open(predicted_path, "w") as f:
                json.dump(predicted_samples, f, indent=4)



if __name__=="__main__":
    main()
