import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import GPT  
import time


class TextDataset(Dataset):
    def __init__(self, file_path, block_size):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            text=text.replace(" ", "").replace("\t", "").replace("\n", "")
        
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    
        data = [self.stoi[ch] for ch in text]
        self.data = torch.tensor(data, dtype=torch.long)
        self.block_size = block_size
        
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y

