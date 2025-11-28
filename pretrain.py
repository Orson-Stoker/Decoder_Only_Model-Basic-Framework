from model import *
from dataset import *
import json
from tqdm import tqdm

class Pretrainer:
    def __init__(self,config_file):
        
        with open(config_file, 'r') as f:
            config= json.load(f)

        self.model_config=config["model"]
        self.train_config=config["train"]

        self.train_dataset=TextDataset(self.train_config["corpus_path"],self.model_config['block_size'])
  
        self.device=self.train_config["device"]
        
        self.model=GPT(
            vocab_size=self.train_dataset.vocab_size,
            embed_size=self.model_config["embed_size"],
            num_heads=self.model_config["num_heads"],
            num_layers=self.model_config["num_layers"],
            block_size=self.model_config["block_size"]
        ).to(self.device)
        self.batch_size=self.train_config["batch_size"]
        self.epochs=self.train_config["epochs"]
        self.save_path=self.train_config["save_path"]
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.optimizer= torch.optim.AdamW(self.model.parameters(), lr=self.train_config["learning_rate"])
        
    def __call__(self): 
        print(f"{int(len(self.train_dataset)/self.batch_size)} items in total.")
        self.model.train()
        for epoch in range(self.epochs):
            total_loss=0
            for x_batch, y_batch in tqdm(self.train_loader,desc="training progress",unit="item"):
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                logits, loss = self.model(x_batch, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss+=loss.item()
            print(f"Average Training Loss:{total_loss:.4f} in Epoch {epoch+1}")

        torch.save(self.model.state_dict(),self.save_path)
        print(f"model was saved at {self.save_path} ")
               
pretrainer=Pretrainer("config.json")
pretrainer()

