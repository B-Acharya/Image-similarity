import torch
from torch.utils.data import Dataset
from torchvision import datasets
# from torchvision.io import  read_image
from PIL import Image

class ImageList(Dataset):

    def __init__(self, image_list, imsize=None, transform=None):
        Dataset.__init__(self)
        self.image_list = image_list
        self.transform = transform
        self.imsize = imsize

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        x = Image.open(self.image_list[i])
        #different method to read images
        x = x.convert("RGB")
        if self.imsize is not None:
            x.thumbnail((self.imsize, self.imsize), Image.ANTIALIAS)
        if self.transform is not None:
            x = self.transform(x)
        return x