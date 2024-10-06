import lightning as L
import mlflow
import os, argparse
#import pytorch_lightning as pl

from datetime import datetime
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning_model import LitConvModel
from mlflow.models import infer_signature
from torch_net import ConvNet
from utils import (
    load_cifar10_loader,
    load_food_loader,
    load_mnist_loader,
    print_auto_logged_info
)

def mnist_experiment(batch_size=256, epochs=10):
    mnist = 'mnist'
    mlflow.set_experiment(mnist)
    mlflow.pytorch.autolog()

    train_loader, eval_loader, test_loader = load_mnist_loader(batch_size)
    batch = iter(train_loader)
    images, labels = next(batch)
    lit_conv_net = LitConvModel(images, labels)
    trainer = L.Trainer(max_epochs = epochs)
    #trainer = pl.Trainer(max_epochs = 10)

    with mlflow.start_run() as run:
        mlflow.set_tag('mlflow.runName', f'mnist_{datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}')
        trainer.fit(lit_conv_net, train_loader, eval_loader)
        trainer.test(lit_conv_net, test_loader)
        print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
        
        model_uri = f"runs:/{run.info.run_id}/model"
        loaded_model = mlflow.pytorch.load_model(model_uri)

        test_batch = iter(test_loader)
        test_images, test_labels = next(test_batch)

        y_pred = loaded_model(test_images)
        signature = infer_signature(test_images.numpy(), y_pred.detach().numpy())
        #NOTE: loaded_model might just be lit_conv_net.
        mlflow.pytorch.log_model(loaded_model, "model", signature=signature)

def cifar10_experiment(batch_size=256, epochs=100):
    mlflow.pytorch.autolog()

    train_loader, eval_loader, test_loader = load_cifar10_loader(batch_size)
    batch = iter(train_loader)
    images, labels = next(batch)
    lit_conv_net = LitConvModel(images, labels)
    """
    NOTE: (callbacks=[EarlyStopping(monitor="val_loss", mode="min")] enables early stopping.
    You can also customize early stopping like the following
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode="max")
    """
    trainer = L.Trainer(callbacks=[EarlyStopping(monitor="val_loss", mode="min")], max_epochs=epochs)
    #trainer = pl.Trainer(max_epochs = 10)

    with mlflow.start_run() as run:
        trainer.fit(lit_conv_net, train_loader, eval_loader)
        trainer.test(lit_conv_net, test_loader)
        print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))

        model_uri = f"runs:/{run.info.run_id}/model"
        loaded_model = mlflow.pytorch.load_model(model_uri)

        test_batch = iter(test_loader)
        test_images, test_labels = next(test_batch)

        y_pred = loaded_model(test_images)
        signature = infer_signature(test_images.numpy(), y_pred.detach().numpy())
        #NOTE: loaded_model might just be lit_conv_net.
        mlflow.pytorch.log_model(loaded_model, "model", signature=signature)

def food_101_experiment(batch_size=256, epochs=10):
    mlflow.pytorch.autolog()
    
    train_loader, eval_loader, test_loader = load_food_loader(batch_size)
    print(f"loaded loaders")
    batch = iter(train_loader)
    print(f"batch: {batch}")
    images, labels = next(batch)
    print(f"images is {images}")

    lit_conv_net = LitConvModel(images, labels)
    trainer = L.Trainer(max_epochs = epochs)
    #trainer = pl.Trainer(max_epochs = 10)

    with mlflow.start_run() as run:
        trainer.fit(lit_conv_net, train_loader, eval_loader)

        print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
        
        model_uri = f"runs:/{run.info.run_id}/model"
        loaded_model = mlflow.pytorch.load_model(model_uri)
        test_batch = iter(test_loader)
        test_images, test_labels = next(test_batch)
        y_pred = loaded_model(test_images)
        signature = infer_signature(test_images.numpy(), y_pred.detach().numpy())
        #NOTE: loaded_model might just be lit_conv_net.
        mlflow.pytorch.log_model(loaded_model, "model", signature=signature)


def main():
    mnist_experiment()
    #cifar10_experiment()
    #food_101_experiment()

if __name__ == "__main__":
    main()
