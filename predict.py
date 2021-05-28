"""Make predictions"""
import torch
import torch.nn as nn
from unet.dataset import Images
from unet.model import UNet
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm
import os
import cv2


def predict(net, data_dir, output_dir, batch_size=4, device='cpu'):
    # Datasets and Dataloaders
    dataset = Images(data_dir, transform=ToTensor())
    dataloader = DataLoader(dataset, batch_size=batch_size)

    if net.n_classes > 1:
        output_fn = nn.Softmax2d()
    else:
        output_fn = nn.Sigmoid()

    i_image = 0
    with tqdm(total=len(dataset), desc=f'Predicting', unit='img') as pbar:
        for i_batch, batch in enumerate(dataloader):
            imgs = batch['image'].to(device=device, dtype=torch.float32)
            preds = net(imgs)

            # output to predictions
            if net.n_classes == 1:
                preds_cls = torch.round(output_fn(preds))
            else:
                preds_cls = torch.argmax(output_fn(preds), dim=1)

            # save predictions
            preds_cls = preds_cls.cpu().detach().numpy()
            for i_sample in range(imgs.shape[0]):
                prediction = preds_cls[i_sample, :, :, :]
                prediction = prediction.reshape(prediction.shape[1:])*255.0
                output_path = os.path.join(output_dir, f"{i_image}.jpg")
                cv2.imwrite(output_path, prediction)
                i_sample += 1

            pbar.update(imgs.shape[0])


if __name__ == '__main__':
    net = UNet(n_classes=1)
    weights_path = './runs/May29_00-41-21_carter-ThinkPad-T460p/checkpoint.pt'
    checkpoint = torch.load(weights_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    data_dir = 'data/membrane/train/image'
    output_dir = 'data/membrane/test_pred'

    predict(net, data_dir, output_dir)