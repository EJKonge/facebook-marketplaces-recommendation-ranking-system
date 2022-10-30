import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn 
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from transformers import BertTokenizer
from transformers import BertModel
from tqdm import tqdm
import pickle



class ImageTextDataSet(torch.utils.data.Dataset):

    def __init__(self, transform: transforms = None, max_len = 100):
        
        self.products = pd.read_csv('Image+Products.csv', lineterminator='\n')
        self.root_dir = 'Image_folder/cleaned_images/'
        self.transform = transform
        self.descriptions = self.products['product_description']
        self.image_id = self.products['image_id']
        self.labels = self.products['category'].to_list()
        self.num_classes = len(set(self.labels))


        self.encoder = {y: x for (x, y) in enumerate(set(self.labels))}
        self.decoder = {x: y for (x, y) in enumerate(set(self.labels))}

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
        self.model.eval()
        self.max_length = max_len
        

        if transform == None:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(128),
                transforms.Resize(128),
                transforms.ToTensor(),
                transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
                ])


        # self.tokenizer = get_tokenizer('basic_english')
        assert len(self.descriptions) == len(self.labels) == len(self.image_id)
    



    def __len__(self):
        return len(self.products)


    def __getitem__(self, index):
        label = self.labels[index]
        label = self.encoder[label]
        label = torch.as_tensor(label)
        image = Image.open(self.root_dir + (self.products.iloc[index, 1] + '.jpg')).convert('RGB')
        image = self.transform(image)
        sentence = self.descriptions[index]
        encoded = self.tokenizer.batch_encode_plus([sentence], max_length=self.max_length, padding='max_length', truncation=True)
        encoded = {key:torch.LongTensor(value) for key, value in encoded.items()}
        with torch.no_grad():
            description = self.model(**encoded).last_hidden_state.swapaxes(1,2)
        
        description = description.squeeze(0)
        return image, description, label



def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
device = get_default_device()




class TextClassifier(torch.nn.Module):
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
                                    nn.Linear(384, 128)).to(device)
        self.decoder = decoder
    def forward(self, inp):
        x = self.main(inp)
        return x


class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        self.features = models.resnet50(pretrained=True).to(device)
        self.text_model = TextClassifier()
        self.main = nn.Sequential(nn.Linear(256, 13))
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
            torch.nn.Linear(512, 128)
            )


    def forward(self, image_features, text_features):
        image_features = self.features(image_features)
        image_features = image_features.reshape(image_features.shape[0], -1)
        text_features = self.text_model(text_features)
        combined_features = torch.cat((image_features, text_features), 1)
        combined_features = self.main(combined_features)
        return combined_features

 
model = CombinedModel()
model.to(device)


dataset = ImageTextDataSet()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32 ,shuffle=True, num_workers=1)


def train_model(model, epochs):

    writer = SummaryWriter()
    print('training model')
    optimiser = optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)
        for i, (image_features, text_features, labels) in tqdm(enumerate(dataloader)):
            model.train()
            num_correct = 0
            num_samples = 0
            image_features = image_features.to(device)
            text_features = text_features.to(device)  # move to device
            labels = labels.to(device)
            predict = model(image_features, text_features)
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
                writer.add_scalar('Training Loss', loss, epoch)
                writer.add_scalar(' Training Accuracy', acc, epoch)
                print('training_loss')
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss:.5f}')
                print(f'Got {num_correct} / {num_samples} with accuracy: {(acc * 100):.2f}%')
                writer.flush()





def check_accuracy(loader, model):
    model.eval()
    print('Checking accuracy on training set')
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for (image_features, text_features, label) in tqdm(loader):
            image_features = image_features.to(device)
            text_features = text_features.to(device)  # move to device
            label = label.to(device)
            predict = model(image_features, text_features)
            _, preds = predict.max(1)
            num_correct += (preds == label).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print(f'Got {num_correct} / {num_samples} with accuracy: {acc * 100}%')
        

if __name__ == '__main__':
    train_model(model, 10)
    model_save_name = 'combined_model.pt'
    path = f"final_models/{model_save_name}" 
    torch.save(model.state_dict(), path)
    with open('combined_model.pkl', 'wb') as f:
        pickle.dump(dataset.decoder, f)
    check_accuracy(dataloader, model)