import os
import argparse
import json

import torch
from fairseq.checkpoint_utils import load_model_ensemble_and_task

import ail2abel # noqa


def preprocess_samples(samples, dictionary_dict):
    preprocessed_samples = []
    for sample in samples:
        preprocessed_sample = {}
        for key, value in sample.items():
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
    checkpoint = "checkpoints/ailabel_x64_O2_270w_100pp/checkpoint_best_new.pt"
    

    os.makedirs(out_dir, exist_ok=True)
    models, saved_cfg, task = load_model_ensemble_and_task([checkpoint])
    model = models[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    for binary_dir in os.listdir(json_dir):
        for func_json_file in os.listdir(os.path.join(json_dir, binary_dir)):
            func_json_path = os.path.join(json_dir, binary_dir, func_json_file)
            predicted_path = os.path.join(out_dir, binary_dir, func_json_file)

            with open(func_json_path, "r") as f:
                samples = json.load(f)
            preprocessed_samples = preprocess_samples(samples, task.dictionary_dict)
            
            mask_idx = task.source_dictionary["ail_token"].index("<mask>")
            for sample, preprocessed_sample in zip(samples, preprocessed_samples):
                preprocessed_sample = {
                    k: v.unsqueeze(0).to(device) for k, v in preprocessed_sample.items()
                }
                masked_code = preprocessed_sample["ail_token"].eq(mask_idx)
                output, _ = model(
                    src_tokens=preprocessed_sample,
                    masked_code=masked_code,
                    classification_head_name="maskvar"
                )
                predicted = torch.argmax(output, dim=-1)
                predicted_tokens = task.source_dictionary["ail_token_label"].string(predicted)


if __name__=="__main__":
    main()
