from torchvision import datasets, transforms
import torch


class DataLoader:
    def __init__(self, path):
        self.path = path

    def change_path(self, path):
        self.path = path

    def load_images_from_folder(self, type):
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

        if type == "train":
            dataset_folder = datasets.ImageFolder(self.path + "/train", transform=transform)
        elif type == "val":
            dataset_folder = datasets.ImageFolder(self.path + "/val", transform=transform)
        else:
            dataset_folder = datasets.ImageFolder(self.path + "/test", transform=transform)

        dataloader = torch.utils.data.DataLoader(dataset_folder, batch_size=64)

        return dataloader


