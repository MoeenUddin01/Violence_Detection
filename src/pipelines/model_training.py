import torch
from src.data.loader import train_loader,test_loader
from src.models.cnn import CNN
from src.models.train import Trainer
from src.models.evaluation import Evaluator
import wandb
import os
from datetime import datetime



#
def main():
    try:
        
        EPOCHS=10
        BATCH_SIZE=16
        LEARNING_RATE=0.001
        DEVICE="cuda" if torch.cuda.is_available() else "cpu"
        
        
        #store configurations in dictionary for logging in WandB
        
        config={
            "epochs":EPOCHS,
            "batch_size":BATCH_SIZE,
            "learning_rate":LEARNING_RATE,
            "device":DEVICE,
            "Model":CNN
        }
        
        #Initialize WandB
        
        wandb.init(
            project="Violence-Detection-CNN",
            config=config
             name=f'Experiment-{datetime.now().strftime("%d_%m_%Y_%H_%M")}'
             
        )
#model initialization
        model=CNN().to(DEVICE)
        print(f"using device={DEVICE}")
        torch.set_default_device(DEVICE)
        
        
        #Initialize Trainer and Evaluator
        
        trainer=Trainer(
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            device=DEVICE,
            model=model,
            model_path="artifacts",
            DEVICE=DEVICE
            
        )
        evaluator=Evaluator(
            batch_size=BATCH_SIZE,
            data=test_loader,
            device=DEVICE,
            model=model,
        )
        
        
