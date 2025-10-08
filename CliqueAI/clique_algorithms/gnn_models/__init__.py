import os

import torch

from .models.scattering_clique.sc_model import ScatteringCliqueModel

GNN_MODEL_FOLDER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models"
)


SC_MODEL = ScatteringCliqueModel(
    checkpoint_path=os.path.join(
        GNN_MODEL_FOLDER_PATH,
        "scattering_clique",
        "packages",
        "checkpoints",
        "scattering_clique",
        "base_model.pt",
    ),
    device="cuda" if torch.cuda.is_available() else "cpu",
)
