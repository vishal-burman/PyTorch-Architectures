import torchvision
from torchvision import transforms

class DataLoaderCIFAR10Classification:
    def __init__(self, train=True):
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=self.transform)

    def return_dataloader(self, batch_size=4, shuffle=False):
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)
