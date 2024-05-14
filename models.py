import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import lightning as L
from torchmetrics import Accuracy, Precision, Recall, F1Score
from modules import DeepBottleneckResNet, VGGBlock
from kan import KANLayer

class EthanNet30(L.LightningModule):
    """
    EthanNet-30 is a deep convolutional neural network designed for image classification tasks, 
    integrating VGG-like blocks and ResNet-like bottleneck blocks for effective feature extraction.
    This model is structured to balance depth and computational efficiency while maintaining high accuracy.

    The network comprises:
    - Three VGG blocks with varying depths and increasing channels, each followed by optional dropout.
    - A DeepBottleneckResNet block that employs bottleneck modules with residual connections to enhance feature propagation without adding excessive parameters.
    - A final pooling layer to reduce spatial dimensions before classification.
    - Two fully connected (FC) layers that condense the feature map into class predictions.
    - Batch normalization and dropout are employed post-pooling to stabilize and regularize the learning process.

    Attributes:
        vgg_block1 (nn.Module): First VGG block starting with 16 channels and 2 repetitions, includes dropout.
        vgg_block2 (nn.Module): Second VGG block with 32 channels and 3 repetitions, includes dropout.
        vgg_block3 (nn.Module): Third VGG block with 64 channels and 3 repetitions, includes dropout.
        resnet_block (DeepBottleneckResNet): ResNet block with layers configured for 256 and 512 channels, each repeated 3 times.
        pool (nn.AvgPool2d): Average pooling layer that reduces the feature map size.
        fc1 (nn.Linear): First fully connected layer reducing dimension to 2048 units.
        bn (nn.BatchNorm1d): Batch normalization for the output of the first FC layer.
        dropout (nn.Dropout): Dropout layer set at 0.5 to prevent overfitting.
        fc2 (nn.Linear): Second fully connected layer that outputs predictions for 10 classes.
        accuracy, precision, recall, f1 (torchmetrics.*): Metric instances for monitoring training performance.

    Methods:
        forward(x): Defines the forward pass of the model.
        on_train_epoch_end(): Handles actions to be performed at the end of each training epoch, such as pruning.
        prune_model(): Prunes the convolutional layers to improve generalization and reduce model size.
        training_step(batch, batch_idx): Processes one batch during training.
        validation_step(batch, batch_idx): Processes one batch during validation.
        test_step(batch, batch_idx): Processes one batch during testing.
        configure_optimizers(): Configures the optimizer used for training.

    Example usage:
        # Assuming `train_dataloader` is defined elsewhere
        model = EthanNet30()
        trainer = L.Trainer()
        trainer.fit(model, train_dataloader)
    """
    def __init__(self):
        super().__init__()
        self.vgg_block1 = VGGBlock(3, 16, 2, dropout=0.2)
        self.vgg_block2 = VGGBlock(16, 32, 3, dropout=0.2)
        self.vgg_block3 = VGGBlock(32, 64, 3, dropout=0.2)
        self.resnet_block = DeepBottleneckResNet(64, [256, 512], [3, 3], dilation=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(512 * 2 * 2, 2048)
        self.bn = nn.BatchNorm1d(2048)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2048, 10)

        # Define the metrics
        self.accuracy = Accuracy(task='multiclass', num_classes=10)
        self.precision = Precision(task='multiclass', num_classes=10)
        self.recall = Recall(task='multiclass', num_classes=10)
        self.f1 = F1Score(task='multiclass', num_classes=10)

    def forward(self, x):
        x = self.vgg_block1(x)
        x = self.vgg_block2(x)
        x = self.vgg_block3(x)
        x = self.resnet_block(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.bn(self.fc1(x))))
        x = self.fc2(x)
        return x
    
    def on_train_epoch_end(self):
        # Pruning is only considered after training epochs, not after validation epochs.
        if (self.current_epoch + 1) % 10 == 0:
            self.prune_model()
            print(f"Applied pruning at the end of epoch {self.current_epoch + 1}")

    def prune_model(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                # Prune based on the absolute value of the weights (magnitude-based pruning)
                prune.l1_unstructured(module, name='weight', amount=0.1)
                # Make pruning permanent
                prune.remove(module, 'weight')

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(torch.argmax(y_hat, dim=1), y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(torch.argmax(y_hat, dim=1), y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = self.accuracy(preds, y)
        precision = self.precision(preds, y)
        recall = self.recall(preds, y)
        f1 = self.f1(preds, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('test_precision', precision, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('test_recall', recall, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('test_f1', f1, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

class EthanNet29K(L.LightningModule):
    """
    EthanNet-29K is a deep convolutional neural network designed for image classification tasks, 
    integrating VGG-like blocks and ResNet-like bottleneck blocks for effective feature extraction.
    This model is structured to balance depth and computational efficiency while maintaining high accuracy.

    The network comprises:
    - Three VGG blocks with varying depths and increasing channels, each followed by optional dropout.
    - A DeepBottleneckResNet block that employs bottleneck modules with residual connections to enhance feature propagation without adding excessive parameters.
    - A final pooling layer to reduce spatial dimensions before classification.
    - A Kolmogorov Arnold Network (KAN) layer that maps the flattened ResNet block output to 10 classes.
    - Batch normalization and dropout are employed post-pooling to stabilize and regularize the learning process.

    Attributes:
        vgg_block1 (nn.Module): First VGG block starting with 16 channels and 2 repetitions, includes dropout.
        vgg_block2 (nn.Module): Second VGG block with 32 channels and 3 repetitions, includes dropout.
        vgg_block3 (nn.Module): Third VGG block with 64 channels and 3 repetitions, includes dropout.
        resnet_block (DeepBottleneckResNet): ResNet block with layers configured for 256 and 512 channels, each repeated 3 times.
        pool (nn.AvgPool2d): Average pooling layer that reduces the feature map size.
        kan (KANLayer): Kolmogorov Arnold Network layer that maps the flattened ResNet block output to 10 classes.
        accuracy, precision, recall, f1 (torchmetrics.*): Metric instances for monitoring training performance.

    Methods:
        forward(x): Defines the forward pass of the model.
        on_train_epoch_end(): Handles actions to be performed at the end of each training epoch, such as pruning.
        prune_model(): Prunes the convolutional layers to improve generalization and reduce model size.
        training_step(batch, batch_idx): Processes one batch during training.
        validation_step(batch, batch_idx): Processes one batch during validation.
        test_step(batch, batch_idx): Processes one batch during testing.
        configure_optimizers(): Configures the optimizer used for training.

    Example usage:
        # Assuming `train_dataloader` is defined elsewhere
        model = EthanNet29K()
        trainer = L.Trainer()
        trainer.fit(model, train_dataloader)
    """
    def __init__(self):
        super().__init__()
        self.vgg_block1 = VGGBlock(3, 16, 2, dropout=0.2)
        self.vgg_block2 = VGGBlock(16, 32, 3, dropout=0.2)
        self.vgg_block3 = VGGBlock(32, 64, 3, dropout=0.2)
        self.resnet_block = DeepBottleneckResNet(64, [256, 512], [3, 3], dilation=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.kan = KANLayer(512 * 2 * 2, 10, k=3)

        # Define the metrics
        self.accuracy = Accuracy(task='multiclass', num_classes=10)
        self.precision = Precision(task='multiclass', num_classes=10)
        self.recall = Recall(task='multiclass', num_classes=10)
        self.f1 = F1Score(task='multiclass', num_classes=10)

    def forward(self, x):
        x = self.vgg_block1(x)
        x = self.vgg_block2(x)
        x = self.vgg_block3(x)
        x = self.resnet_block(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x, _, _, _ = self.kan(x)  # Unpacking all returned values
        return x
    
    def on_train_epoch_end(self):
        # Pruning is only considered after training epochs, not after validation epochs.
        if (self.current_epoch + 1) % 10 == 0:
            self.prune_model()
            print(f"Applied pruning at the end of epoch {self.current_epoch + 1}")

    def prune_model(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                # Prune based on the absolute value of the weights (magnitude-based pruning)
                prune.l1_unstructured(module, name='weight', amount=0.1)
                # Make pruning permanent
                prune.remove(module, 'weight')

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(torch.argmax(y_hat, dim=1), y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(torch.argmax(y_hat, dim=1), y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = self.accuracy(preds, y)
        precision = self.precision(preds, y)
        recall = self.recall(preds, y)
        f1 = self.f1(preds, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('test_precision', precision, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('test_recall', recall, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('test_f1', f1, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
