from model import GabinModel
from gabin_data_loader import GabinSpectrogramDataset, ToTensor
import torch 
from torch import nn
import numpy as np
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    liobject = ["sound 0", "sound 1", "sound 2", "sound 3", "sound 4", "sound 5", "sound 6", "sound 7", "sound 8", "sound 9"]
    transform = transforms.Compose([ToTensor()])

    trainset = GabinSpectrogramDataset(".", transforms = transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    testset = GabinSpectrogramDataset(".", status="test", transforms = transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    element= next(iter(testloader))
    print(element["image"].shape)
    img_show(element["image"], liobject[element['cat'][0]])


    model  = GabinModel()

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

        for elemen in trainloader:
            data, target = elemen["image"], elemen["cat"]

            optimizer.zero_grad()

            output = model(data.float())

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