#import pytorch_lightning as pl
import lightning as L

from torch import nn
from torch.optim import Adam
from torch_net import ConvNet 
from torchmetrics import Accuracy

class LitConvModel(L.LightningModule):
    def __init__(self, X, y, lr=0.01):
        super().__init__()
        self.model = ConvNet(X, y)
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.model.num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss)
        acc = self.val_accuracy(logits, y)
        self.log("val_accuracy", acc)
        #pred = logits.argmax(dim=1)
        #acc = self.val_accuracy(pred, y)
        #self.log("val_accuracy_mlflow", acc)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

#class LitConvModel(pl.LightningModule):
#    def __init__(self, X, y, lr=0.01):
#        super().__init__()
#        self.model = ConvNet(X, y)
#        self.loss_fn = nn.CrossEntropyLoss()
#        self.lr = lr
#
#    def forward(self, x):
#        return self.model(x)
#
#    def training_step(self, batch, batch_idx):
#        x, y = batch
#        output = self(x)
#        loss = self.loss_fn(output, y)
#        self.log("train_loss", loss)
#        return loss
#
#    def validation_step(self, batch, batch_idx):
#        x, y = batch
#        output = self(x)
#        loss = self.loss_fn(output, y)
#        self.log("val_loss", loss)
#        return loss
#
#    def configure_optimizers(self):
#        return Adam(self.parameters(), lr=self.lr)
