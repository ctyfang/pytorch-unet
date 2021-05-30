import hydra
import pytorch_lightning as pl
import torchvision

from project.metrics import *
from project.model import *


class LitUNet(pl.LightningModule):
    def __init__(self, input_channels, n_classes, n_prefeats=64, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # Network Configuration
        self.pre_encoder = SequentialConv(input_channels, n_prefeats, 2, 3)
        self.pre_down = nn.MaxPool2d(2, (2, 2))
        self.encoder = Encoder(levels=3)

        n_bottleneck_feats = int(n_prefeats * (2 ** (self.encoder.levels + 1)))
        self.bottleneck = SequentialConv(n_bottleneck_feats // 2,
                                         n_bottleneck_feats)

        self.decoder = Decoder(levels=4)

        n_output_feats = int(n_bottleneck_feats * (0.5 ** (self.decoder.levels)))
        self.output_conv = nn.Conv2d(n_output_feats, n_classes, kernel_size=1)

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
        self.log('loss/train', loss.item(), on_step=True, on_epoch=False,
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
        if self.hparams.n_classes == 1:
            output = torch.round(output)
        else:
            output = torch.argmax(output, dim=1)

        self.log('loss/val', loss.item(), on_step=False, on_epoch=True)
        self.log('acc/val', self.val_f1(output, y), on_step=False, on_epoch=True)

        if batch_idx == 0:
            img_grid = torchvision.utils.make_grid(x)
            label_grid = torchvision.utils.make_grid(y)
            pred_grid = torchvision.utils.make_grid(output.type(torch.int))

            self.logger.experiment.add_image('input', img_grid, self.current_epoch)
            self.logger.experiment.add_image('predictions', pred_grid, self.current_epoch)
            self.logger.experiment.add_image('label', label_grid, self.current_epoch)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch['image']
        y_hat = self(x)
        output = self.output_fn(y_hat)

        # Generate predictions
        if self.n_classes == 1:
            output = torch.round(output)
        else:
            output = torch.argmax(output, dim=1)
        return {'image': x, 'prediction': output}

    def test_epoch_end(self, test_outputs):
        x_arr, output_arr = [], []
        for out in test_outputs:
            x, output = out['image'], out['prediction']
            output = output.type(torch.int)
            x_arr.append(x)
            output_arr.append(output)

        x = torch.cat(x_arr, dim=0)
        output = torch.cat(output_arr, dim=0)
        img_grid = torchvision.utils.make_grid(x)
        pred_grid = torchvision.utils.make_grid(output)

        self.logger.experiment.add_image('test_input', img_grid, 0)
        self.logger.experiment.add_image('test_predictions', pred_grid, 0)
        self.logger.experiment.flush()

    def configure_optimizers(self):
        # Optimization
        if self.hparams.n_classes > 1:
            self.loss_fn = nn.CrossEntropyLoss()
            self.output_fn = nn.Softmax2d()
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()
            self.output_fn = nn.Sigmoid()

        return hydra.utils.instantiate(self.hparams.optimizer, params=self.parameters())

from project.dataset import Images
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # # Training
    # validation_ratio = 0.1
    # batch_size = 1
    # data_dir = 'data/membrane/train'
    #
    # dataset = UNetDataset(data_dir,
    #                       transform=ToTensor(), target_transform=ToTensor())
    # n_train, n_val = int(dataset.size * (1 - validation_ratio)), int(
    #     dataset.size * validation_ratio)
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset,
    #                                                            [n_train, n_val])
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
    #                               shuffle=True, num_workers=8)
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
    #                             shuffle=False, num_workers=8)
    #
    # unet = LitUNet()
    # trainer = pl.Trainer(gpus=1)
    # trainer.fit(unet, train_dataloader, val_dataloader)

    # Testing
    unet = LitUNet.load_from_checkpoint('../lightning_logs/version_8/checkpoints/epoch=22-step=620.ckpt')
    trainer = pl.Trainer(gpus=1)
    test_dataset = Images(root_dir='../data/membrane/test/', transform=ToTensor())
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=8, shuffle=False)
    trainer.test(unet, test_dataloaders=test_dataloader)
