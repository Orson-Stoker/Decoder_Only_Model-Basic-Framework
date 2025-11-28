import torch,json
from model import *
class Vocaburary():
    def __init__(self,file_path):

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            text=text.replace(" ", "").replace("\t", "").replace("\n", "")

        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self,text):
        return torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long).unsqueeze(0)
    
    def decode(self,idx):
        return "".join([self.itos[i] for i in idx.tolist()[0]])


class Evaluator:
    def __init__(self,config_file):
        with open(config_file, 'r') as f:
            config= json.load(f)
        
        self.eval_config=config["eval"]
        self.model_config=config["model"]
        self.vocab=Vocaburary(self.eval_config["corpus_path"])
        self.model=GPT(
            vocab_size=self.vocab.vocab_size,
            embed_size=self.model_config["embed_size"],
            num_heads=self.model_config["num_heads"],
            num_layers=self.model_config["num_layers"],
            block_size=self.model_config["block_size"]
        )
        self.max_new_tokens=self.eval_config["max_new_tokens"]
        self.model.load_state_dict(torch.load(self.eval_config["model_path"]))

    def inference(self):
        while 1:
            text=input("You:")
            encoded_text=self.vocab.encode(text)
            encoded_output=self.model.generate(encoded_text,self.max_new_tokens)
            output=self.vocab.decode(encoded_output)
            print(f"Model:{output}")

eval=Evaluator("config.json")
eval.inference()

