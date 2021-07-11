import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from .utils import get_image_classification_dataset, image_interpolation


class DatasetCIFAR10Classification(Dataset):
    def __init__(self, resize=224, train=True, upsample_size=None):
        self.tuple_list = get_image_classification_dataset(train=train)
        self.transforms = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.upsample_size = upsample_size

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
        if self.upsample_size is not None:
            images = image_interpolation(images, target_size=self.upsample_size)
        labels = torch.tensor(labels)
        return {
            "pixel_values": images,
            "labels": labels,
        }


class DataLoaderCIFAR10Classification:
    def __init__(self, resize=256, train=True, upsample_size=None):
        self.dataset = DatasetCIFAR10Classification(
            resize=resize, train=train, upsample_size=upsample_size
        )

    def return_dataloader(self, batch_size=4, shuffle=False):
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.dataset.collate_fn,
        )
