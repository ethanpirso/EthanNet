import torch
import torchvision
from torchvision.transforms import v2
from models import EthanNet30, EthanNet29K
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from strategies import CustomDDPStrategy

def load_data():
    # Transformations for the CIFAR-10 dataset
    transform = v2.Compose(
        [v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.AutoAugment(torchvision.transforms.autoaugment.AutoAugmentPolicy.CIFAR10)
        ])

    # Downloading and loading the CIFAR-10 training dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(trainset, [train_size, val_size])

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, persistent_workers=True, pin_memory=True)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8, persistent_workers=True, pin_memory=True)

    # Downloading and loading the CIFAR-10 test dataset
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=8, persistent_workers=True, pin_memory=True)

    return trainloader, valloader, testloader

def train_model(trainloader, valloader, testloader):
    early_stop_callback = EarlyStopping(monitor="val_loss",
                                        min_delta=0.01,
                                        patience=10,
                                        verbose=False,
                                        mode="min")
    trainer = L.Trainer(max_epochs=80,
                        # strategy=CustomDDPStrategy(),
                        accelerator='gpu', 
                        devices=1,
                        # precision="bf16-true",
                        callbacks=[early_stop_callback],
                        deterministic=True)
    with trainer.init_module():
        model = EthanNet29K()
        model.to(torch.float32)
    trainer.fit(model, trainloader, valloader)
    trainer.test(model, testloader)
