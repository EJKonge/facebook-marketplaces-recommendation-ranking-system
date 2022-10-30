from transformers import BertTokenizer
from transformers import BertModel
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import pickle
import torch.nn as nn
from tqdm import tqdm


class TextDataset(Dataset):
    def __init__(self, max_length: int= 100):
        super().__init__()
        self.products = pd.read_csv('Image+Products.csv', lineterminator='\n')
        self.labels = self.products['category'].to_list()
        self.descriptions = self.products['product_description']
        self.num_classes = len(set(self.labels))
        self.encoder = {y: x for (x, y) in enumerate(set(self.labels))}
        self.decoder = {x: y for (x, y) in enumerate(set(self.labels))}

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
        self.model.eval()
        self.max_length = max_length
        


    def __len__(self):
        return len(self.descriptions)


    def __getitem__(self, idx):
        label = self.labels[idx]
        label = self.encoder[label]
        label = torch.as_tensor(label)
        sentence = self.descriptions[idx]
        encoded = self.tokenizer.batch_encode_plus([sentence], max_length=self.max_length, padding='max_length', truncation=True)
        encoded = {key:torch.LongTensor(value) for key, value in encoded.items()}
        with torch.no_grad():
            description = self.model(**encoded).last_hidden_state.swapaxes(1,2)
        
        description = description.squeeze(0)

        return description, label




dataset = TextDataset()

class CNN(torch.nn.Module):
    def __init__(self,
                 input_size: int = 768,
                 num_classes: int = 13,
                 decoder: dict = None):
        super().__init__()
        self.main = nn.Sequential(nn.Conv1d(input_size, 256, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=2, stride=2),
                                    nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=2, stride=2),
                                    nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=2, stride=2),
                                    nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Flatten(),
                                    nn.Linear(384 , 128),
                                    nn.ReLU(),
                                    nn.Linear(128, num_classes))
        self.decoder = decoder
    

    
    def forward(self, inp):
        x = self.main(inp)
        return x


# sets dataloader variables
validation_split = 0.15
batch_size =32
shuffle_dataset = True
random_seed = 42


dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]


# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_samples = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler, drop_last=True)
val_samples = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler, drop_last=True)

def train_model(model, epochs):
    writer = SummaryWriter()
    model.train()
    print('training model')
    optimiser = optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        for phase in [train_samples, val_samples]:
            if phase == train_samples:
                print('training')
                model.train()
            else:
                model.eval()
                print('val')

            for i, (features, labels) in tqdm(enumerate(phase)):
                if phase == 'train':
                    torch.set_grad_enabled(phase)
                num_correct = 0
                num_samples = 0
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

                if i % 130 == 129:
                    print(f'Epoch {epoch + 1}/{epochs}')
                    print('-' * 10)
                    if phase == train_samples:
                        writer.add_scalar('Training Loss', loss, epoch)
                        writer.add_scalar(' Training Accuracy', acc, epoch)
                        print('training_loss')
                        print(f'Loss: {loss:.4f} Acc: {acc*100:.1f}%')
                        print(f'Got {num_correct} / {num_samples} with accuracy: {acc * 100}%')
                    else:
                        writer.add_scalar('Validation Loss', loss, epoch)
                        writer.add_scalar('Validation Accuracy', acc, epoch)
                        print('val_loss') 
                        print(f'Loss: {loss:.4f} Acc: {acc*100:.1f}%')
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
            scores = model(feature)
            _, preds = scores.max(1)
            num_correct += (preds == label).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print(f'Got {num_correct} / {num_samples} with accuracy: {acc * 100}%')

if '__name__" == __main__':
    cnn = CNN()
    print(dataset[0])
    train_model(cnn, 10)
    model_save_name = 'text_model.pt'
    path = f"final_models/{model_save_name}" 
    torch.save(cnn.state_dict(), path)
    with open('text_decoder.pkl', 'wb') as f:
        pickle.dump(dataset.decoder, f)
    check_accuracy(train_samples, cnn)
    check_accuracy(val_samples, cnn)