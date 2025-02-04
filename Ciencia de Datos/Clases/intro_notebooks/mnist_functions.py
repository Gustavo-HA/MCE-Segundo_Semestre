import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output, x    # return x for visualization

def get_train_data():
    train_data = datasets.MNIST(
        root = 'data',
        train = True,                         
        transform = ToTensor(), 
        download = True,            
    )
    return train_data

def get_train_features(model):
    train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor(), 
    download = True,            
    )

    loaders = {
        'train' : torch.utils.data.DataLoader(train_data, 
                                              batch_size=100, 
                                          shuffle=True, 
                                          num_workers=1),   
    #    'test'  : torch.utils.data.DataLoader(test_data, 
    #                                          batch_size=100, 
    #                                          shuffle=True, 
    #                                          num_workers=1),
    }

    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        j = 0
        for images, labels in loaders['train']:
            test_output, last_layer = model(images)            
            if j > 0:
              all_embeddings = np.append(all_embeddings, last_layer.numpy(), axis = 0)
              all_labels = np.append(all_labels, labels.numpy(), axis = 0)
            else:
              all_embeddings = last_layer.numpy()
              all_labels = labels
            j += 1
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
    return all_embeddings, all_labels

def get_pca(embeddings, ncomp=2):
    scaler = StandardScaler()
    pca = PCA(n_components=ncomp)

    # Ajustamos en datos de entrenamiento
    scaler.fit(embeddings)
    train_img = scaler.transform(embeddings)

    # Obtenemos las representaciones
    x_pca = pca.fit_transform(train_img)
    
    return scaler, pca, x_pca


def get_pca_pixels(train_data, img_width, img_height, nsample = 10000, ncomp=2):
    scaler = StandardScaler()
    pca = PCA(n_components=ncomp)
    
    nmax = nsample
    temp_loader = torch.utils.data.DataLoader(train_data, batch_size=nmax, shuffle=True, num_workers=2)
    xx, yy = next(iter(temp_loader))
    
    X = xx.squeeze().numpy()
    X = X.reshape(nmax, img_width*img_height)
    y = yy.numpy()
    
    # Ajustamos en datos de entrenamiento
    scaler.fit(X)
    train_img = scaler.transform(X)
    
    # Obtenemos las representaciones
    x_pca = pca.fit_transform(train_img)
    
    return scaler, pca, x_pca, y

class simple_linear_classifier(nn.Module):
  def __init__(self, input_dim=28*28, output_dim=10):
    super(simple_linear_classifier, self).__init__()
    self.linear = nn.Linear(input_dim, output_dim)

  def forward(self, x):
    x = x.view(x.size(0), -1)
    x = self.linear(x)
    return x


                
                
              