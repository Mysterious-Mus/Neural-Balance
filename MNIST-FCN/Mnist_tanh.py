import argparse
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
import os
from sklearn.model_selection import train_test_split
import numpy as np
import random
from tqdm import tqdm
import csv

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MNIST:
    def __init__(self, batchsize, seed, frac=1, test=False, norm=True):
        # Define transformations
        # transforms.ToTensor() already converts the data to a tensor and scales it to [0, 1]
        transform_list = [transforms.ToTensor()]
        if norm:
            # Now normalize around the mean 0.5 and std 0.5 after scaling to [0, 1]
            transform_list.append(transforms.Normalize((0.5,), (0.5,)))
        
        transform = transforms.Compose(transform_list)

        # Load CIFAR10 data with transforms
        mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        # Extract labels for stratification
        if frac < 1:

            print('getting a fraction of the dataset')

            labels = [label for _, label in mnist_train]

            _, subset_indices = train_test_split(
                range(len(mnist_train)),
                test_size=frac,
                random_state=seed,
                stratify=labels
            )
            mnist_train = Subset(mnist_train, subset_indices)

        if not test:
            self.dataset = DataLoader(mnist_train, batch_size=batchsize, shuffle=True)
        else:
            self.dataset = DataLoader(mnist_test, batch_size=batchsize, shuffle=False)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.relu = nn.Tanh()
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_size[0])])
        for i in range(len(hidden_size)-1):
            self.layers.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
        self.layers.append(nn.Linear(hidden_size[len(hidden_size)-1], num_classes))
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
        x = self.layers[-1](x)
        x = self.relu(x)
        return x
    
def neuralBalance(inl, oul, order):
    incoming = torch.linalg.norm(inl.weight, dim=1, ord=order)
    outgoing = torch.linalg.norm(oul.weight, dim=0, ord=order)
    optimal_l = torch.sqrt(outgoing/incoming)
    inl.weight.data *= optimal_l.unsqueeze(1)
    oul.weight.data /= optimal_l

def main():
    parser = argparse.ArgumentParser(description='Mnist')
    parser.add_argument("--epochs", required=False, default = 100, type=int,help="Number Of Epochs The Model Is Trained For")
    parser.add_argument("--lr", required=False, default=1e-3, type=float, help="constant learning rate for model")
    parser.add_argument("--model", required=False, default = 'small_fcn', type=str, choices = ['small_fcn', 'medium_fcn', 'large_fcn', 'xlarge_fcn'], help="choose dataset")
    parser.add_argument("--dataset", required=False, default = 'mnist', type=str, choices = ['mnist'],help="choose dataset")
    parser.add_argument("--gpu", required=False, default = '0', type=str,help="Choose GPU to use")
    parser.add_argument("--batchsize", required=False, default = 256, type=int,help="Choose batch_size for the dataset")
    parser.add_argument("--l2_weight", required=False, default = 0, type=float, help="Multiplier for L2 Regularizer")
    parser.add_argument("--seed", required=False, default = 42, type=int,help="Choose seed")
    parser.add_argument("--neural_balance", required=False, default = 0, type=int,help="Whether we train with neural balance or not")
    parser.add_argument("--neural_balance_epoch", required=False, default = 1, type=int,help="Every how many epochs we are doing neural balance.")
    parser.add_argument("--order", required=False, default = 2, type=int,help="Order of norm when doing neural balance.")
    parser.add_argument("--neuralFullBalanceAtStart", required = False, default = 0, type = int, help="Whether neural balance is fully performed before the model's training begins")
    parser.add_argument("--trainDataFrac", required = False, default = 1, type = float, help = "What fraction of the training dataset is used in training")
    parser.add_argument("--reversed", required = False, default = 0, type = int, help = "Reversed neural Balancing")
    parser.add_argument("--foldername", required = True, default = "default", type=str, help = "folder name")
    parser.add_argument("--filename", required = True, default = "default", type=str, help = "file name")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    # device = torch.device(f"cpu" if torch.cuda.is_available() else "cpu")

    trainloader = MNIST(batchsize=args.batchsize, test=False, seed=args.seed, frac=args.trainDataFrac).dataset
    testloader = MNIST(batchsize=args.batchsize, test=True, seed=args.seed).dataset

    model = None

    if args.model == 'small_fcn':
        model = NeuralNet(784, [256], 10).to(device)
    elif args.model == 'medium_fcn':
        model = NeuralNet(784, [256, 128], 10).to(device)
    elif args.model == 'large_fcn':
        model = NeuralNet(784, [512, 256, 128, 64], 10).to(device)
    elif args.model == 'xlarge_fcn':
        model = NeuralNet(784, [4096, 2048, 1024, 512, 256, 128, 64, 32, 16], 10).to(device)

    criterion = nn.CrossEntropyLoss()

    if args.l2_weight > 0:
        print(f"using l2 = {args.l2_weight}")
    if args.order == 2:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.l2_weight)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr)

    hist = {}
    hist['train_loss'] = []
    hist['test_loss'] = []
    hist['test_acc'] = []

    lay = list(model.layers.children())

    if args.neuralFullBalanceAtStart == 1:
        print('full balancing at start')
        while(True):
            restart=False
            if reversed != 0:
                for i in reversed(range(len(lay)-1)):
                    lay1, lay2 = lay[i], lay[i+1]
                    incoming = torch.linalg.norm(lay1.weight, dim=1, ord=args.order)
                    outgoing = torch.linalg.norm(lay2.weight, dim=0, ord=args.order)
                    optimal_l = torch.sqrt(outgoing/incoming).sum()/incoming.shape[0]
                    print(optimal_l)
                    if optimal_l > 1.001 or optimal_l < .999:
                        restart=True
                    neuralBalance(lay1, lay2, order = args.order)
            else:
                for i in range(len(lay)-1):
                    lay1, lay2 = lay[i], lay[i+1]
                    incoming = torch.linalg.norm(lay1.weight, dim=1, ord=args.order)
                    outgoing = torch.linalg.norm(lay2.weight, dim=0, ord=args.order)
                    optimal_l = torch.sqrt(outgoing/incoming).sum()/incoming.shape[0]
                    print(optimal_l)
                    if optimal_l > 1.001 or optimal_l < .999:
                        restart=True
                    neuralBalance(lay1, lay2, order = args.order)
            if not restart:
                break

    for epoch in range(args.epochs):
        train_loss = 0
        for i, (images, labels) in tqdm(enumerate(trainloader)):
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            
            if args.l2_weight > 0 and args.order == 1:
                params = torch.cat([x.view(-1) for x in model.parameters()])
                l1_regularization = args.l2_weight * torch.norm(params, 1)
                loss+= l1_regularization
            train_loss+=loss


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{args.epochs}], Loss: {train_loss / len(trainloader):.4f}')
            
        hist['train_loss'].append(train_loss / len(trainloader))

        if args.neural_balance == 1 and epoch % args.neural_balance_epoch == 0:
            print()
            print('performing neural balance')
            print()
            lay = list(model.layers.children())
            if reversed != 0:
                for i in reversed(range(len(lay)-1)):
                    neuralBalance(lay[i], lay[i+1], order = args.order)
            else:
                for i in range(len(lay)-1):
                    neuralBalance(lay[i], lay[i+1], order = args.order)

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            test_loss = 0
            for images, labels in testloader:
                images = images.reshape(-1, 28*28).to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                test_loss += loss.item()

            avg_loss = test_loss / len(testloader)

            hist['test_loss'].append(avg_loss)
            hist['test_acc'].append(100 * correct / total)

            print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')
            print(f'Loss of the network on the 10000 test images: {avg_loss:.4f}%')

if __name__ == "__main__":
    main()
