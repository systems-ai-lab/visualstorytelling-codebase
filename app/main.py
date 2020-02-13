import torch
from dataset import VIST
from torchvision import transforms
from PIL import Image

val_transform = transforms.Compose([
        transforms.Resize(224, interpolation=Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])


dataset = VIST(dataset_dir='../dataset/', vocabulary_treshold=5, type='val', 
            transform= val_transform)

data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, 
                shuffle=True, num_workers=0, collate_fn=dataset.collate_fn)


for bi, (image_stories, targets_set, lengths_set, photo_squence_set, 
    album_ids_set) in enumerate(data_loader):
        print(bi)