# dataloader

implemete data loader for pytorch


## Basic Usage

install
```bash
pip install  git+https://github.com/GitDataAI/jz_dataloader.git    
```

## example

```py
import torch
import io
from torch.utils.data import DataLoader
from jz_dataloader import ImageDataset, JiaozifsDataset
from torchvision import transforms
import numpy as np
from PIL import Image
import shutil

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()

    trans = transforms.Compose([
        np.float32,
        transforms.ToTensor()
    ])

    dataset = ImageDataset(
        "jimmy",
        "tiny-imagenet",
        "bbeb6af9acc0d9e3bb31c0fc5a348ed1",
        "749a5e9867ee61ddc152c7fa40c2f3e8",
        type="wip",
        path="val",
        refName="main",
        transform=transforms.ToTensor(),
    )
    print("init ready")
    # 创建数据加载器
    batch_size = 32
    shuffle = True
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print("loader ready")
    # 迭代数据加载器
    for inputs, labels in dataloader:
        print(f"inputs shape {inputs.shape()} labels shape {labels}")

    print("ready")
```