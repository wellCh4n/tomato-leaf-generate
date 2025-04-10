import torch
from datasets import load_dataset


class TomatoLeafDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        dataset = load_dataset("wellCh4n/tomato-leaf-disease-image").filter(lambda example, idx: example['label'] == 0,
                                                                            with_indices=True)
        self.dataset = dataset['train'].with_format('torch')
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]['image']
        if self.transform:
            img = self.transform(img)
        return img

