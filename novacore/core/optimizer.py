
import torch
def create_optimizer(model, lr=3e-4):
    return torch.optim.AdamW(model.parameters(), lr=lr)
