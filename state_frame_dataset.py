import json
import glob

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

class PairDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()

        self.data_path = data_path
        self.files = glob.glob(f"{data_path}/json/*.json")
        self.transform = v2.Compose([
            #v2.ColorJitter(brightness=0.2, hue=0.2),
            #v2.Resize(112),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]

        with open(file, "r") as f:
            data = json.load(f)

        state = torch.as_tensor(data["state"][-1])

        step_id = data["step_id"]
        frame = [Image.open(f"{self.data_path}/img/{step_id}_{i}.png") for i in range(2)]
        frame = self.transform(frame)
        frame = torch.concat(frame)

        return state, frame

def get_dataloader(task):
    data_path = f"pair_data/{task}"
    dataset = PairDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=16)

    return dataloader