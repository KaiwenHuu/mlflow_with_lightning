import torchvision.transforms as transforms

from mlflow import MlflowClient
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST

def load_mnist_loader(batch_size):
    dataset = MNIST(
            root='./data',
            train=True,
            transform=transforms.ToTensor(),
            download=True
        )
    test_set = MNIST(
            root='./data',
            train=False,
            transform=transforms.ToTensor(),
            download=True
        )

    train_set, eval_set = random_split(dataset, [50000, 10000])
    train_loader = DataLoader(
            dataset=train_set, 
            batch_size=batch_size, 
            shuffle=True
        )
    eval_loader = DataLoader(
            dataset=eval_set,
            batch_size=batch_size,
            shuffle=True
        )
    test_loader = DataLoader(
            dataset=test_set,
            batch_size=batch_size,
            shuffle=True
        )
    return train_loader, eval_loader, test_loader

def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print(f"run_id: {r.info.run_id}")
    print(f"artifacts: {artifacts}")
    print(f"params: {r.data.params}")
    print(f"metrics: {r.data.metrics}")
    print(f"tags: {tags}")
