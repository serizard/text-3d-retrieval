import numpy as np
import torch
import open_clip

def normalize_pc(pc):
    pc = pc - np.mean(pc, axis=0)
    if np.max(np.linalg.norm(pc, axis=1)) < 1e-6:
        pc = np.zeros_like(pc)
    else:
        pc = pc / np.max(np.linalg.norm(pc, axis=1))
    return pc

@torch.no_grad()
def process_input(user_input, open_clip_model, device):
    with torch.no_grad():
        text = open_clip.tokenizer.tokenize(user_input).to(device)
        return open_clip_model.encode_text(text).cpu().numpy()