import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from .utils import get_image_classification_dataset, image_interpolation


class DatasetCIFAR10Classification(Dataset):
    def __init__(self, resize=224, train=True):
        self.tuple_list = get_image_classification_dataset(train=train)
        assert resize <= 224, "resize should be less than or equal to 224"
        self.transforms = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(
        self,
    ):
        return len(self.tuple_list)

    def __getitem__(self, idx):
        tuple_idx = self.tuple_list[idx]
        return tuple_idx

    def collate_fn(self, batch):
        images = []
        labels = []
        for sample in batch:
            images.append(Image.open(sample[0]))
            labels.append(sample[1])
        if self.transforms is not None:
            images = [self.transforms(image) for image in images]
        images = torch.stack(images)
        labels = torch.tensor(labels)
        return {
            "pixel_values": images,
            "labels": labels,
        }

def DataLoaderCIFAR10Classification(resize: int = 224,
                                    train: bool = True,
                                    batch_size: int = 4,
                                    shuffle: bool = False):
    """
    Returns DataLoader for CIFAR10 classification
    """
    dataset = DatasetCIFAR10Classification(resize=resize, train=train)
    return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=dataset.collate_fn,
            )
