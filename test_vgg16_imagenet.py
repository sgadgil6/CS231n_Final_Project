import numpy as np
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils, models, datasets
import torch.nn as nn
from torch import optim, cuda
import os
from torchsummary import summary
from tqdm import tqdm

DATA_DIR = '/home/groups/kpohl/CS231n_data/imagenet/imagenet_images'
SAVE_DATA_DIR = './data/imagenet'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
print("Using device {}".format(device))
BATCH_SIZE = 128
checkpoint_pth = "/home/groups/kpohl/CS231n_data/imagenet/models/vgg16_best_model_imagenet.pth"


# Class to create read imagenet images
class ImageFolderWithImageName(datasets.ImageFolder):
    """
    Dataset which returns tuple (img, target, img_id)
    """

    def __getitem__(self, index):
        sample, target = super(ImageFolderWithImageName, self).__getitem__(index)
        path = self.imgs[index][0]
        # print(path.split('/')[-1][:-4])

        return sample, target, path.split('/')[-1][:-4]


# Data Augmentation
image_transforms = {
    # Train uses data augmentation
    'train':
        transforms.Compose([
            transforms.ToPILImage(),  
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),  
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])  # Imagenet standards
            # transforms.RandomAffine(degrees=(0, 5), translate=(0.02, 0.08), scale=(1.0, 1.1)),
            # transforms.RandomHorizontalFlip(),
            # transforms.CenterCrop(224),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.0], std=[1.0])
        ]),
    # Validation does not use augmentation
    'test':
        transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # transforms.Normalize(mean=[0.0], std=[1.0])
        ]),
}


def get_pretrianed_model(model_name, num_classes):
    if model_name == 'vgg_16':
        model = models.vgg16(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False
        n_inputs = model.classifier[6].in_features

        model.classifier[6] = nn.Sequential(nn.Linear(n_inputs, 256),
                                            nn.ReLU(),
                                            nn.Dropout(0.2),
                                            nn.Linear(256, num_classes),
                                            nn.LogSoftmax(dim=1))
        model = model.to(device)
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        
        n_inputs = model.fc.in_features

        model.fc = nn.Sequential(nn.Linear(n_inputs, 256),
                                            nn.ReLU(),
                                            nn.Dropout(0.3),
                                            nn.Linear(256, num_classes),
                                            nn.LogSoftmax(dim=1))
        model = model.to(device)
    else:
        raise NotImplementedErrors

    return model

dataset = ImageFolderWithImageName(DATA_DIR, transform=transforms.Compose([transforms.ToTensor()]))
# print(dataset.classes)
train_dataset, test_dataset = random_split(dataset, lengths=[45000, len(dataset) - 45000])

dataloaders = {"train": DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True),
               "test": DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)}

# ImageNet has 596 labels
vgg16_model = get_pretrianed_model('vgg_16', len(dataset.classes))
resnet_model = get_pretrianed_model('resnet18', len(dataset.classes))
# print(summary(vgg16_model, input_size=(3, 224, 224), batch_size=BATCH_SIZE))
print("Size of Train Set = {}".format(len(dataloaders['train'].dataset)))
print("Size of Test Set = {}".format(len(dataloaders['test'].dataset)))

criterion = nn.NLLLoss()
optimizer = optim.Adam(vgg16_model.parameters())


def train(model, criterion, optimizer, train_loader, val_loader, n_epochs=50, print_every=1):
    max_train_acc = 0
    max_val_acc_top5 = float('-inf')
    history = []
    correct_labels_top1 = []
    incorrect_labels_top1 = []
    correct_labels_top5 = []
    incorrect_labels_top5 = []
    model.optimizer = optimizer

    for epoch in range(n_epochs):
        model.train()
        val_acc_top1 = 0
        train_acc_top1 = 0
        train_acc_top5 = 0
        val_acc_top5 = 0
        train_loss = 0
        val_loss = 0
        correct_labels_top1 = []
        incorrect_labels_top1 = []
        correct_labels_top5 = []
        incorrect_labels_top5 = []
        img_ids_val_list = []
        # Training
        print("Training Epoch {}".format(epoch))
        for x_train, y_train, img_ids_train in tqdm(train_loader):
            # print(x_train.shape)
            # print(y_train.shape)
            # print(len(img_ids_train))
            for i in range(x_train.shape[0]):
                x_train[i] = image_transforms['train'](x_train[i])
            x_train = x_train.to(device)
            y_train = y_train.to(device)

            labels = y_train
            optimizer.zero_grad()
            output = model(x_train)
            labels = labels.long()
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_train.size(0)

            _, pred_top5 = output.topk(k=5, dim=1)
            pred_top5 = pred_top5.t()
            correct_top5 = pred_top5.eq(labels.view(1, -1).expand_as(pred_top5))
            # top-5 Accuracy
            train_acc_top5 += correct_top5[:5].view(-1).float().sum(0, keepdim=True)
            # top-1 Accuracy
            train_acc_top1 += correct_top5[:1].view(-1).float().sum(0, keepdim=True)

        # Validating
        with torch.no_grad():

            was_training = model.training
            model.eval()

            print("Validating Epoch {}".format(epoch))
            for x_val, y_val, img_ids_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)

                labels = y_val
                output = model(x_val)
                labels = labels.long()
                loss = criterion(output, labels)
                val_loss += loss.item() * x_val.size(0)

                _, pred_top5 = output.topk(k=5, dim=1)

                y_val = y_val.cpu().detach().numpy()
                for i in range(x_val.shape[0]):
                    img_ids_val_list.append(img_ids_val[i])
                    if labels[i] in pred_top5[i]:
                        val_acc_top5 += 1
                        correct_labels_top5.append(y_val[i])
                        if labels[i] == pred_top5[i, 0]:
                            val_acc_top1 += 1
                            correct_labels_top1.append(y_val[i])
                        else:
                            incorrect_labels_top1.append(y_val[i])
                    else:
                        incorrect_labels_top1.append(y_val[i])
                        incorrect_labels_top5.append(y_val[i])

            if was_training: #back to training
                model.train()

        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)

        train_acc_top1 = train_acc_top1 / len(train_loader.dataset)
        val_acc_top1 = val_acc_top1 / len(val_loader.dataset)
        train_acc_top5 = train_acc_top5 / len(train_loader.dataset)
        val_acc_top5 = val_acc_top5 / len(val_loader.dataset)
        
        train_acc_top1 = train_acc_top1.cpu().detach().numpy()
        train_acc_top5 = train_acc_top5.cpu().detach().numpy()

        # If we get better val accuracy, save the labels for analysis, and save the model
        if val_acc_top5 > max_val_acc_top5:
            max_val_acc_top5 = val_acc_top5
            np.save(os.path.join(SAVE_DATA_DIR, "correct_labels_top5_imagenet.npy"), correct_labels_top5)
            np.save(os.path.join(SAVE_DATA_DIR, "incorrect_labels_top5_imagenet.npy"), incorrect_labels_top5)
            np.save(os.path.join(SAVE_DATA_DIR, "img_ids_imagenet.npy"), img_ids_val_list)
            torch.save(model.state_dict(), checkpoint_pth)

        if (epoch + 1) % print_every == 0:
            print(
                'Epoch: %d, Train Loss: %0.4f, Val Loss: %0.4f, Train Acc Top 1/5: %0.4f %0.4f, '
                'Val Acc Top1/5: %0.4f %0.4f' % (epoch, train_loss,
                                                 val_loss,
                                                 train_acc_top1,
                                                 train_acc_top5,
                                                 val_acc_top1,
                                                 val_acc_top5))

        history.append([train_loss, val_loss, train_acc_top1, val_acc_top1, train_acc_top5, val_acc_top5])
        np.save(os.path.join(DATA_DIR, "history.npy"), history)
    return history, correct_labels_top1, correct_labels_top5, incorrect_labels_top1, incorrect_labels_top5


history, correct_labels_top1, correct_labels_top5, incorrect_labels_top1, incorrect_labels_top5 = train(vgg16_model, criterion, optimizer, dataloaders['train'], dataloaders['test'])
