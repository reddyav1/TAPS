import argparse
import time
import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import lightning.pytorch as pl
import torchvision.models as models
from tqdm import tqdm

from folder import ImageFolder
from utils import AverageMeter, accuracy


class ResNetModel(nn.Module):
    def __init__(self, num_classes=12):
        super(ResNetModel, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if args.ik_init else None)
        self.model.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.model(x)
    
    
def create_transforms(cropped=True, small=False, model_type='resnet'):
    if model_type == 'vit':
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        
    if small:
        train_transform = transforms.Compose([
        # TODO: experiment with transforms
                transforms.Resize(72),
                transforms.CenterCrop(72),
                transforms.ToTensor(),
                normalize,
        ])

        test_transform = transforms.Compose([
        # TODO: experiment with transforms
                transforms.Resize(72),
                transforms.CenterCrop(72),
                transforms.ToTensor(),
                normalize,
        ])
    else:
        if cropped:
            test_transform = transforms.Compose([
                transforms.Resize([224,224]),
                transforms.ToTensor(),
                normalize,
            ])
            train_transform = transforms.Compose([
                transforms.Resize([224,224]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else: 
            test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
            ])

            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
    return train_transform, test_transform
    
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainset_dir', type=str, default="/export/r32/data/visda17/train")
    parser.add_argument('--dataset_frac', type=float, default=1.0)
    parser.add_argument('--val_split', type=float, default=0.15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--ik_init', action='store_true')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--gpu_idx', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='model.pth')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()

    # Get the transforms
    train_transform, test_transform = create_transforms(cropped=True, small=False)
    # Create the dataset
    fullset = ImageFolder(args.trainset_dir, transform=train_transform, frac=args.dataset_frac)

    val_size = int(args.val_split * len(fullset))
    train_size = len(fullset) - val_size
    train_set, val_set = random_split(fullset, [train_size, val_size])

    # Create the dataloader
    trainloader = DataLoader(train_set, 
                             batch_size=args.batch_size, 
                             num_workers=args.num_workers,
                             shuffle=True,
                             pin_memory=True)
    
    valloader = DataLoader(val_set, 
                             batch_size=args.batch_size, 
                             num_workers=args.num_workers,
                             shuffle=False,
                             pin_memory=True)

    net = ResNetModel(num_classes=12).cuda(device=args.gpu_idx)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.wd,
                                )
    # scheduler = torch.optim.lr_scheduler.MultiStep

    epoch_bar = tqdm(range(args.epochs), desc="Overall Training", position=0)
    for i, epoch in enumerate(epoch_bar):
        batch_bar = tqdm(trainloader, desc="Current Epoch", position=1, leave=False)
        train_loss_meter = AverageMeter('train_loss')
        net.train()
        for j, batch in enumerate(batch_bar):

            x, y = batch
            optimizer.zero_grad()

            # Pass inputs through network
            outputs = net(x.cuda(args.gpu_idx))

            # Compute loss and its gradients
            loss = loss_fn(outputs, y.cuda(args.gpu_idx))
            loss.backward()

            # Update the weights
            optimizer.step()

            train_loss_meter.update(loss.item(), x.size(0))
            if j % 20 == 0:
                batch_bar.set_postfix(t_loss=f"{train_loss_meter.avg:.3f}")

            # update tqdm progress bar and log to tensorboard

        # validation routine
        net.eval()
        batch_bar_val = batch_bar = tqdm(valloader, desc="Validation", position=1, leave=False)
        val_loss_meter = AverageMeter('val_loss')
        val_acc_meter = AverageMeter('val_acc')
        with torch.no_grad():
            for j, batch_val in enumerate(batch_bar_val):
                x_val, y_val = batch
                outputs_val = net(x_val.cuda(args.gpu_idx))
                loss_val = loss_fn(outputs_val, y_val.cuda(args.gpu_idx))
                val_loss_meter.update(loss_val.item(), x_val.size(0))

                val_acc = accuracy(outputs_val.cpu(), y_val.cpu())
                val_acc_meter.update(val_acc[0].item(), x_val.size(0))
                
        epoch_bar.set_postfix(val_loss=f"{val_loss_meter.avg:.3f}",
                              val_acc=f"{val_acc_meter.avg:.1f}")

        torch.save(net.state_dict(), args.save_path)

