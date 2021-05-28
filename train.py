import torch
from torch import optim
import torchvision
import torch.nn as nn
from unet.dataset import UNetDataset
from unet.model import UNet
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os


def validate(net, n_validate, loss_fn, output_fn, dataloader, device='cpu'):
    """Return average metrics over the validation set"""
    net.eval()

    with tqdm(total=n_validate, desc=f'Validation', unit='img') as pbar:

        total_loss, total_correct, total_pixels = 0.0, 0, 0
        for i_batch, batch in enumerate(dataloader):
            imgs = batch['image'].to(device=device, dtype=torch.float32)
            labels = batch['label'].to(device=device, dtype=torch.float32)

            preds = net(imgs)
            if preds.shape[2:] != labels.shape[2:]:
                labels = nn.functional.interpolate(labels, preds.shape[2:],
                                                   mode='nearest')
            loss = loss_fn(preds, labels)

            # Compute number correct
            if net.n_classes == 1:
                preds_cls = torch.round(output_fn(preds))
            else:
                preds_cls = torch.argmax(output_fn(preds), dim=1)

            # Accumulate metrics
            total_loss += loss.item()
            total_correct += (preds_cls == labels).sum().item()
            total_pixels += preds.shape[0]*preds.shape[2]*preds.shape[3]

            pbar.set_postfix(**{'loss (batch)': loss.item()})
            pbar.update(imgs.shape[0])

    return total_loss, total_correct/total_pixels


def train(net, data_dir, validation_ratio,
          epochs, batch_size, learning_rate,
          device='cpu'):

    # Datasets and Dataloaders
    dataset = UNetDataset(data_dir,
                          transform=ToTensor(), target_transform=ToTensor())
    n_train, n_val = int(dataset.size * (1-validation_ratio)), int(dataset.size * validation_ratio)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset,
                                                               [n_train, n_val])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Optimizer and Loss Function
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate,
                              weight_decay=1e-8, momentum=0.9)
    if net.n_classes > 1:
        loss_fn = nn.CrossEntropyLoss()
        output_fn = nn.Softmax2d()
    else:
        loss_fn = nn.BCEWithLogitsLoss()
        output_fn = nn.Sigmoid()

    # Training Loop
    writer = SummaryWriter()
    for i_epoch in range(epochs):
        net.train()

        with tqdm(total=n_train, desc=f'Epoch {i_epoch+1}/{epochs}', unit='img') as pbar:

            epoch_loss = 0
            for i_batch, batch in enumerate(train_dataloader):
                imgs = batch['image'].to(device=device, dtype=torch.float32)
                labels = batch['label'].to(device=device, dtype=torch.float32)

                preds = net(imgs)
                if preds.shape[2:] != labels.shape[2:]:
                    labels = nn.functional.interpolate(labels, preds.shape[2:], mode='nearest')
                loss = loss_fn(preds, labels)
                epoch_loss += loss

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i_batch == 0:
                    # output to predictions
                    if net.n_classes == 1:
                        preds_cls = torch.round(output_fn(preds))
                    else:
                        preds_cls = torch.argmax(output_fn(preds), dim=1)

                    # create grid of images
                    img_grid = torchvision.utils.make_grid(imgs)
                    pred_grid = torchvision.utils.make_grid(preds_cls)
                    label_grid = torchvision.utils.make_grid(labels)

                    # write to tensorboard
                    writer.add_image('inputs', img_grid, i_epoch)
                    writer.add_image('pred_masks', pred_grid, i_epoch)
                    writer.add_image('true_masks', label_grid, i_epoch)

                pbar.update(imgs.shape[0])

        # Validate
        loss_val, acc_val = validate(net, n_val, loss_fn, output_fn,
                                     val_dataloader, device=device)

        # Tensorboard Logging
        writer.add_scalar('acc/val', acc_val, i_epoch)
        writer.add_scalar('loss/val', loss_val, i_epoch)
        writer.add_scalar('loss/train', epoch_loss, i_epoch)
        writer.flush()

        # Checkpoint
        torch.save({
            'epoch': i_epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(writer.log_dir, 'checkpoint.pt'))


if __name__ == '__main__':
    net = UNet(n_classes=1)
    data_dir = 'data/membrane/train'
    epochs = 30
    learning_rate = 1e-4
    batch_size = 30
    validation_ratio = 0.1

    train(net, data_dir, validation_ratio, epochs, batch_size, learning_rate)