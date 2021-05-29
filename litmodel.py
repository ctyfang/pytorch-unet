import pytorch_lightning as pl
import torchvision
import torchmetrics

from metrics import *
from model import *


class LitUNet(pl.LightningModule):
    def __init__(self, n_classes=1, n_prefeats=64, learning_rate=1e-4):
        super().__init__()
        # Network Configuration
        self.n_classes = n_classes
        self.n_prefeats = n_prefeats

        self.pre_encoder = SequentialConv(1, n_prefeats, 2, 3)
        self.pre_down = nn.MaxPool2d(2, (2, 2))
        self.encoder = Encoder(levels=3)

        self.n_bottleneck_feats = int(
         n_prefeats * (2 ** (self.encoder.levels + 1)))
        self.bottleneck = SequentialConv(self.n_bottleneck_feats // 2,
                                      self.n_bottleneck_feats)

        self.decoder = Decoder(levels=4)

        self.n_output_feats = int(
        self.n_bottleneck_feats * (0.5 ** (self.decoder.levels)))
        self.output_conv = nn.Conv2d(self.n_output_feats, self.n_classes, kernel_size=1)

        # Optimization
        if self.n_classes > 1:
            self.loss_fn = nn.CrossEntropyLoss()
            self.output_fn = nn.Softmax2d()
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()
            self.output_fn = nn.Sigmoid()
        self.learning_rate = learning_rate

        # Validation Metrics
        self.train_f1 = DiceCoefficient()
        self.val_f1 = DiceCoefficient()

    def forward(self, x):
        x = self.pre_down(self.pre_encoder(x))
        skip_connections = [x]

        x, x_intermed = self.encoder(x)
        skip_connections += x_intermed

        x = self.bottleneck(x)

        x = self.decoder(x, skip_connections)
        return self.output_conv(x)

    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        y_hat = self(x)

        if y_hat.shape[2:] != y.shape[2:]:
            y = nn.functional.interpolate(y, y_hat.shape[2:],
                                          mode='nearest')
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss.item(), on_step=True, on_epoch=True,
                 prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        y_hat = self(x)

        if y_hat.shape[2:] != y.shape[2:]:
            y = nn.functional.interpolate(y, y_hat.shape[2:],
                                          mode='nearest')
        loss = self.loss_fn(y_hat, y)
        output = self.output_fn(y_hat)

        # Generate predictions
        if self.n_classes == 1:
            output = torch.round(output)
        else:
            output = torch.argmax(output, dim=1)

        self.log('val_loss', loss.item())
        self.log('val_acc', self.val_f1(output, y),
                 on_step=True, on_epoch=False)

        if batch_idx == 0:
            img_grid = torchvision.utils.make_grid(x)
            label_grid = torchvision.utils.make_grid(y)
            pred_grid = torchvision.utils.make_grid(output.type(torch.int))

            self.logger.experiment.add_image('input', img_grid, self.current_epoch)
            self.logger.experiment.add_image('predictions', pred_grid, self.current_epoch)
            self.logger.experiment.add_image('label', label_grid, self.current_epoch)
        return loss

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=self.learning_rate,
                                   weight_decay=1e-8, momentum=0.9)

from dataset import UNetDataset
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

if __name__ == '__main__':
    validation_ratio = 0.1
    batch_size = 30
    data_dir = 'data/membrane/train'

    dataset = UNetDataset(data_dir,
                          transform=ToTensor(), target_transform=ToTensor())
    n_train, n_val = int(dataset.size * (1 - validation_ratio)), int(
        dataset.size * validation_ratio)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset,
                                                               [n_train, n_val])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=8)

    unet = LitUNet()
    trainer = pl.Trainer()
    trainer.fit(unet, train_dataloader, val_dataloader)
