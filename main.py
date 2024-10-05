import lightning as L
import mlflow
import os, argparse
#import pytorch_lightning as pl

from lightning_model import LitConvModel
from mlflow.models import infer_signature
from torch_net import ConvNet
from utils import (
    load_mnist_loader,
    print_auto_logged_info
)

def mnist_experiment():
    mlflow.pytorch.autolog()
    batch_size = 256

    train_loader, eval_loader, test_loader = load_mnist_loader(batch_size)
    batch = iter(train_loader)
    images, labels = next(batch)
    convnet = ConvNet(images, labels)
    lit_conv_net = LitConvModel(images, labels)
    trainer = L.Trainer(max_epochs = 10)
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
    #raise NotImplementedError()

if __name__ == "__main__":
    main()
