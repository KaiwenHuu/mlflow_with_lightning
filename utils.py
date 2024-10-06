import torchvision.transforms as transforms

from mlflow import MlflowClient
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST, Food101, CIFAR10

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
    dataset_size = len(dataset)
    train_size = int(dataset_size*0.8)
    train_set, eval_set = random_split(dataset, [train_size, dataset_size-train_size])
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

def load_cifar10_loader(batch_size):
    dataset = CIFAR10(
        root='./data',
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )
    test_set = CIFAR10(
        root='./data',
        train=False,
        transform=transforms.ToTensor(),
        download=True
    )
    dataset_size = len(dataset)
    train_size = int(dataset_size*0.8)
    train_set, eval_set = random_split(dataset, [train_size, dataset_size-train_size])
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

def load_food_loader(batch_size, resize=255, center_crop=224):

    train_transforms = transforms.Compose(
        [
            #transforms.RandomRotation(30),
            #transforms.RandomResizedCrop(224),
            #transforms.RandomHorizontalFlip(),ImageNetPolicy(),
            transforms.Resize(resize),
            transforms.CenterCrop(center_crop),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ]
    )
    test_transforms = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(center_crop),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ]
    )

    dataset = Food101(
        root='./data',
        #split='train',
        #transform=transforms.ToTensor(),
        transform=train_transforms,
        download=True
    )
    test_set = Food101(
        root='./data',
        split='test',
        transform=transforms.ToTensor(),
        download=True
    )
    dataset_size = len(dataset)
    train_size = int(dataset_size*0.8)
    train_set, eval_set = random_split(dataset, [train_size, dataset_size-train_size])
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
