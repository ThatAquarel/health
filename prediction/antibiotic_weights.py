import torch

from antibiotic_usage_train import AntibioticPredictor, MODEL


model = AntibioticPredictor()
model.load_state_dict(torch.load(MODEL))
model.eval()

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
...
