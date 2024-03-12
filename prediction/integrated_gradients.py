import torch

from captum.attr import IntegratedGradients
from antibiotic_usage_train import AntibioticPredictor, MODEL


model = AntibioticPredictor()
model.load_state_dict(torch.load(MODEL))
model.eval()


ig = IntegratedGradients(model)
attributions, approximation_error = ig.attribute(
    (input1, input2),
    baselines=(baseline1, baseline2),
    method="gausslegendre",
    return_convergence_delta=True,
)
