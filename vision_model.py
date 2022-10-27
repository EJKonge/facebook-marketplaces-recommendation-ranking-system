import torch
from torch import Tensor
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models, datasets
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pickle
from PIL import Image
from PIL import ImageFile

class ImagesDataset(torch.utils.data.Dataset):

    def __init__(self, transform: transforms = None):
        self.products = pd.read_csv('Image+Products.csv', lineterminator='\n')
        self.root_dir = 'Image_folder/cleaned_images/'
        self.transform = transform
        self.image_id = self.products['image_id']
        self.labels = self.products['category'].to_list()
        self.num_classes = len(set(self.labels))

        self.encoder = {y: x for (x, y) in enumerate(set(self.labels))}
        self.decoder = {x: y for (x, y) in enumerate(set(self.labels))}

        if transform == None:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(64),
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
                ])
        assert len(self.labels) == len(self.image_id)

    def __len__(self):
        return len(self.products)

    def __getitem__(self, index):
        label = self.labels[index]
        label = self.encoder[label]
        label = torch.as_tensor(label)
        image = Image.open(self.root_dir + self.image_id[index] + '_cleaned.jpg').convert('RGB')
        image = self.transform(image)

        return image, label

dataset = ImagesDataset()

train_split = 0.7
validation_split = 0.15
batch_size = 32

data_size = len(dataset)
print(f'dataset contains {data_size} Images')

train_size = int(train_split * data_size)
val_size = int(validation_split * data_size)
test_size = data_size - (val_size + train_size)
train_data, val_data, test_data = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

train_samples = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_samples = DataLoader(val_data, batch_size=batch_size)
test_samples = DataLoader(test_data, batch_size=batch_size)


class CNN(nn.Module):
    def __init__(self, decoder: dict = None):
        super(CNN, self).__init__()
        self.features = models.resnet50(pretrained=True).to(device)
        self.decoder = decoder
        for i, param in enumerate(self.features.parameters()):
            if i < 47:
                param.requires_grad=False
            else:
                param.requires_grad=True
        self.features.fc = nn.Sequential(
            nn.Linear(2048, 1024), # first arg is the size of the flattened output from resnet50
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 13)
            )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.shape[0], -1)
        return x


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
device = get_default_device()


model = CNN()
model.to(device)


def train(model, epochs):
    writer = SummaryWriter()
    model.train()
    print('training model')
    optimiser = optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)
        for phase in [train_samples, val_samples]:
            if phase == train_samples:
                print('training')
            else:
                print('val')
                
            for i, (features, labels) in enumerate(phase):
                num_correct = 0
                num_samples = 0
                features, labels = features, labels
                features = features.to(device)  
                labels = labels.to(device)
                predict = model(features)
                labels = labels
                loss = F.cross_entropy(predict, labels)
                _, preds = predict.max(1)
                num_correct += (preds == labels).sum()
                num_samples += preds.size(0)
                acc = float(num_correct) / num_samples
                loss.backward()
                optimiser.step()
                optimiser.zero_grad()

                if i % 10 == 9:
                    if phase == train_samples:
                      writer.add_scalar('Training Loss', loss, epoch)
                      writer.add_scalar(' Training Accuracy', acc, epoch)
                      print('training_loss')
                    else:
                      writer.add_scalar('Validation Loss', loss, epoch)
                      writer.add_scalar('Validation Accuracy', acc, epoch)
                      print('val_loss') 
                    # print(batch) # print every 50 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss}')
                    print(f'Got {num_correct} / {num_samples} with accuracy: {acc * 100}%')
                    writer.flush()

def check_accuracy(loader, model):
    model.eval()
    if loader == train_samples:
        print('Checking accuracy on training set')
    else:
        print('Checking accuracy on evaluation set')
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for feature, label in loader:
            feature = feature.to(device)  # move to device
            label = label.to(device)
            scores = model(feature)
            _, preds = scores.max(1)
            num_correct += (preds == label).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print(f'Got {num_correct} / {num_samples} with accuracy: {acc * 100}%')


if '__name__" == __main__':
    train(model, 10)
    model_save_name = 'image_model.pt'
    path = f"final_models/{model_save_name}" 
    torch.save(model.state_dict(), path)
    with open('image_decoder.pkl', 'wb') as f:
            pickle.dump(dataset.decoder, f)
    check_accuracy(train_samples, model)
    check_accuracy(val_samples, model)