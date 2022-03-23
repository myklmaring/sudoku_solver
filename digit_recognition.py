import torchvision
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import torch
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import Dataset
import argparse
import os

""" define and train convolutional neural network for digit recognition """

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.relu = nn.ReLU(inplace=True)

        # Encoder
        self.conv1 = ConvLayer(1, 32, kernel_size=7, stride=1)
        self.conv2 = ConvLayer(32, 32, kernel_size=5, stride=1)
        self.cv2fc = nn.Conv2d(32, 64, kernel_size=28, stride=1)

        self.conv = nn.Sequential(
            self.conv1,
            self.relu,
            self.conv2,
            self.relu,
            self.cv2fc,
            self.relu
        )

        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 10)
        self.softmax = nn.Softmax(dim=3)

        self.fc = nn.Sequential(
            self.fc1,
            self.relu,
            self.fc2,
            self.softmax
        )

    def forward(self, x):
        x = x.to(self.device)
        x = self.conv(x)    # output of conv layer is (batch, channel, 1, 1)
        output = self.fc(x.permute(0, 2, 3, 1))
        return output

class ConvLayer(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride):
        super(ConvLayer, self).__init__()

        # pad to maintain output H, W
        pad = (kernel_size - 1) // 2
        self.pad1 = nn.ReflectionPad2d(pad)

        # Convolutional Layer
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride)

    def forward(self, x):
        output = self.pad1(x)
        output = self.conv1(output)
        return output

class MNIST19(Dataset):
    def __init__(self, mnist_data, transform=None, target_transform=None):
        nonzero_inds = np.where(mnist_data.targets != 0)

        self.img_labels = mnist_data.targets[nonzero_inds]
        self.imgs = mnist_data.data[nonzero_inds]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.imgs[None, idx].float()
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


def train(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.Resize(args.img_size),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.mul(255))])

    transform1 = transforms.Compose([transforms.Resize(args.img_size),
                                    transforms.ToTensor(),
                                    transforms.GaussianBlur(3, 0.5),
                                    transforms.RandomAffine(5, translate=(0.10, 0.10), scale=(0.7, 1.3), shear=10),
                                    transforms.Lambda(lambda x: x.mul(255))])


    mnist_data_train = torchvision.datasets.MNIST(args.datapath, transform=transform)
    mnist19_train = MNIST19(mnist_data_train)

    mynetwork = MyNetwork().to(device)
    optimizer = Adam(mynetwork.parameters(), float(args.lr))
    mylossfn = nn.CrossEntropyLoss()

    for epoch in range(int(args.epochs)):
        data_loader = torch.utils.data.DataLoader(mnist19_train, batch_size=int(args.batch_size), shuffle=True)

        iter_bar = tqdm(data_loader)

        myloss = 0
        nbatches = len(iter_bar)

        for images, labels in iter_bar:
            labels = labels.to(device)
            pred = mynetwork(images)
            loss = mylossfn(pred.squeeze(), labels.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            myloss += loss.item()

        if args.print is True:
            print("Epoch {} Loss: {}".format(epoch, myloss/nbatches))

    if args.save:
        if 'model' not in os.listdir():
            os.mkdir('model')
        ckpt_model_path = "model/mnist_model_epoch_" + str(args.epochs) + ".pth"
        torch.save(mynetwork.state_dict(), ckpt_model_path)

    return


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.Resize(args.img_size),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.mul(255))])

    # load test MNIST set
    mnist_data_test = torchvision.datasets.MNIST(args.datapath, train=False, transform=transform)
    mnist19_test = MNIST19(mnist_data_test)
    data_loader = torch.utils.data.DataLoader(mnist19_test, batch_size=1, shuffle=True)

    iter_bar = tqdm(data_loader)

    # load trained model
    model = MyNetwork().to(device)
    model.load_state_dict(torch.load(args.test_model))
    model.eval()

    correct = 0

    for image, label in iter_bar:
        image = image.to(device)
        label = label.to(device)
        pred_onehot = model(image)
        pred = torch.argmax(pred_onehot.squeeze(), dim=0)   # assumes batch is size=1

        if pred == label:
            correct += 1

    print("Percent of correct predictions was {:.2f}".format(100 * correct/mnist19_test.__len__()))

    return





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--datapath", type=str, help="path to MNIST dataset")
    parser.add_argument("--batch-size", default=16, help="batch size of mnist images for training")
    parser.add_argument("--img-size", default=28, help="pixel height/width of the mnist images")
    parser.add_argument("--epochs", default=5, help="number of training epochs")
    parser.add_argument("--lr", default=0.0001, help="learning rate")
    parser.add_argument("--save", default='yes', help="save network if yes")
    parser.add_argument("--print", default=False, help="print average epoch training loss")
    parser.add_argument("--test-model", default=None, help="name of trained model to evaluate")

    args = parser.parse_args()

    train(args)
    test(args)
