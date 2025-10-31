# datasets/pokemon_diffusion_data.zip
'''
ls datasets/pokemon_diffusion_data>>
    images  
    metadata.jsonl

ls datasets/pokemon_diffusion_data/images/ | head
    0001_bulbasaur.jpg
    0002_ivysaur.jpg
    0003_venusaur.jpg
    0004_charmander.jpg
    0005_charmeleon.jpg

head pokemon_diffusion_data/metadata.jsonl>>
    {"file_name": "0001_bulbasaur.jpg", "text": "A pok...ith this POK\u00e9MON.", "types": ["grass", "poison"], "height": 7, "weight": 69}
    {"file_name": "0002_ivysaur.jpg", "text": "A pokem...the ability to stand on its hind legs.", "types": ["grass", "poison"], "height": 10, "weight": 130}
'''

# INSERT_YOUR_CODE

import os
import json
import random
import zipfile
from torch.utils.data import Dataset

class PokemonDiffusionDataset(Dataset):
    """
    直接从 zip 文件中读取 images 和 metadata.jsonl，实现训练/测试划分和按需提取图片路径与文本描述
    """
    def __init__(self, zip_path, split='train', split_ratio=0.9, transform=None, random_seed=42):
        """
        :param zip_path: 路径，如 'datasets/pokemon_diffusion_data.zip'
        :param split: 'train' or 'test'
        :param split_ratio: 训练集比例
        :param transform: 图像预处理方法
        :param random_seed: 数据子集划分随机种子
        """
        self.zip_path = zip_path
        self.transform = transform
        self.entries = []  # list of dict: {'file_name', 'text', ...}
        self._img_dir = "images"
        self._read_metadata(split, split_ratio, random_seed)

    def _read_metadata(self, split, split_ratio, random_seed):
        # 1. 打开 zip，读取 metadata.jsonl
        all_entries = []
        with zipfile.ZipFile(self.zip_path, "r") as zipf:
            with zipf.open('metadata.jsonl') as metafile:
                for line in metafile:
                    data = json.loads(line.decode("utf-8"))
                    all_entries.append(data)
        
        random.seed(random_seed)
        random.shuffle(all_entries)
        n_train = int(len(all_entries) * split_ratio)
        if split == "train":
            self.entries = all_entries[:n_train]
        else:
            self.entries = all_entries[n_train:]

    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):
        entry = self.entries[idx]
        file_name = entry["file_name"]
        text = entry["text"]

        # 返回 (zip 内部图片二进制, 文本, img路径)
        # 只返回图片路径和文本描述
        img_path_in_zip = os.path.join(self._img_dir, file_name)
        result = {
            "img_path_in_zip": img_path_in_zip,
            "text": text,
            "file_name": file_name,
            "entry": entry
        }
        return result

def test_pokemon_diffusion_dataset():
    dataset = PokemonDiffusionDataset("datasets/pokemon_diffusion_data.zip", split="train")
    print(f"Train Size: {len(dataset)}")
    sample = dataset[0]
    print(sample)
    dataset_test = PokemonDiffusionDataset("datasets/pokemon_diffusion_data.zip", split="test")
    print(f"Test Size: {len(dataset_test)}")


