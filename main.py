import os
import torch
import lightning as L
from utils import load_data, train_model

if __name__ == '__main__':
    # Set the MASTER_ADDR and MASTER_PORT environment variables
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8888'
    print("Starting Distributed Training with MASTER_ADDR:", os.environ['MASTER_ADDR'])
    print("MASTER_PORT:", os.environ['MASTER_PORT'])

    # Set the default precision for matrix multiplication
    torch.set_float32_matmul_precision('medium')

    # Set the seed for reproducibility
    L.seed_everything(seed=42, workers=True)

    # Load the data
    trainloader, valloader, testloader = load_data()

    # Train the model
    try:
        train_model(trainloader, valloader, testloader)
    except Exception as e:
        print("An error occurred during model training:", str(e))
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            