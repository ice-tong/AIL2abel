import torch


src_pt = "checkpoints/ailabel_x64_O2_270w_100pp/checkpoint_best.pt"
dst_pt = "checkpoints/ailabel_x64_O2_270w_100pp/checkpoint_best_new.pt"

state = torch.load(src_pt, map_location=torch.device("cpu"))

state["cfg"]["model"]._name = "ail2abel"

for k, v in list(state["model"].items()):
    if k.startswith("encoder.lm_value"):
        state["model"].pop(k)
        print("pop", k)
    if k.startswith("encoder.sentence_encoder.mem_combine"):
        state["model"].pop(k)
        print("pop", k)

# breakpoint()

torch.save(state, dst_pt)

