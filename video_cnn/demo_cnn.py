import torch 
import cv2
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

liobject = ["T-shirt/top", "Trouser",  "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

def img_show(image, title):
    plt.imshow(image.numpy()[0][0], cmap = 'gray')
    plt.title(title)
    plt.show()

def image_error(image, output, label):
    fig, axs = plt.subplots(2)
    fig.suptitle(liobject[label])
    axs[0].imshow(image.numpy()[0][0], cmap = 'gray')
    axs[1].bar(liobject, output.detach()[0].numpy())
    plt.show()

class CNN_model(nn.Module):
    def __init__(self):
        super(CNN_model, self).__init__()

        #Input channels = 1 (nombre de dimension  des donnees d'entrees dans notre cas on est enz noir et blanc donc une seulle dimension)
        #Output chanels = 3 (nombre de dimmension des donnees de sortie, le nombre de fois que seront apliquee des filtres convolutif a notre image)
        self.conv1 = nn.Conv2d(1, 3, kernel_size = 3, stride=1, padding =1)

        #une couche maxpool
        self.pool = nn.MaxPool2d(kernel_size = 2,  stride = 2, padding = 0)

        #partie lineaire de notre CNN
        self.fc1 = nn.Linear(3*14*14, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        
        #10 sorties
    
    def forward(self, x):
        #dimension de x, (1, 28,28) -> (3, 28,28)
        x = self.conv1(x)

        #dimmension de x, (3, 28, 28) -> (3, 28, 28)
        x = F.relu(x)

        #dimmension de x, (3, 28, 28) -> (3, 14, 14)
        x = self.pool(x)

        #debut du reseau lineaire associe

        #lineariastion de x
        #dimmension de x, (3, 28, 28) -> 3*15*15 = 675(vecteur)
        x =  x.view(-1, 3*14*14)

        #dimmension de x, 675 -> 128
        x = F.relu(self.fc1(x))

        #dimmension de x, 128 -> 64
        x = F.relu(self.fc2(x))

        #dimmension de x, 64 -> 10
        x = F.relu(self.fc3(x))
        
        return x



if __name__ == "__main__":

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    image, label = next(iter(testloader))
    
    img_show(image, liobject[label[0].item()])


    model  = CNN_model()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    #entraimement


    # nombre de tour d'entrainement du model
    n_epochs = 5

    valid_loss_min = np.Inf # erreur sur le jeu de validation( test set)

    for epoch in range(1, n_epochs+1):

        train_loss = 0.0
        
        #model en entrainement
        model.train()

        for data, target in trainloader:
            optimizer.zero_grad()

            output = model(data)

            loss = criterion(output, target)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()*data.size(0)

        train_loss = train_loss/len(trainloader.sampler)
            
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch, train_loss))

    model.eval()

    image, label = next(iter(testloader))

    output = model(image)

    loss = criterion(output, label)

    image_error(image, output, label[0].item())
