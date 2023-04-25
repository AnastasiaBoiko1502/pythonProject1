import torch
import csv
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import os
import matplotlib.pyplot as plt
import numpy as np
import dvc.api

params = dvc.api.params_show()
split_number = params['split']['number']

if not os.path.exists('data'):
    os.mkdir('data')

data_folder_path = os.path.abspath('data')
print(data_folder_path)

# создание папки для частей данных
if not os.path.exists('data/train_chunks'):
    os.mkdir('data/train_chunks')

transform = transforms.Compose([
    transforms.ToTensor(),
    # другие преобразования
])

train_dataset = datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
print(len(train_dataset))

def collate_fn(batch):
    """
    Функция для объединения нескольких элементов в пакет.

    Преобразует объекты PIL.Image.Image в тензоры.
    """
    # Разделение данных и меток
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]

    # Преобразование PIL.Image.Image в тензоры
    data = [transform(img) for img in data]

    return [data, target]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)


chunk_sizes = [10000] * split_number  # размер каждой части
train_chunks = random_split(train_dataset, chunk_sizes)

for i, chunk in enumerate(train_chunks):
    # сохранение каждой части в отдельную папку внутри папки 'data'
    chunk_folder_path = os.path.join(data_folder_path, 'train_chunks')
    chunk_path = os.path.join(chunk_folder_path, f'train_chunk_{i}.pt')
    torch.save(chunk, chunk_path)

for i in range(split_number):
    file_path = os.path.join(data_folder_path, 'train_chunks', f"train_chunk_{i}.pt")
    with open(file_path, 'rb') as f:
        chunk = torch.load(f)
        print(f"Chunk {i} size: {len(chunk)}")

# Создание файла csv
with open('data/data_filenames.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Запись заголовка
    writer.writerow(['filename'])
    # Запись названия каждого файла
    for i in range(split_number):
        filename = f"train_chunks/train_chunk_{i}.pt"
        writer.writerow([filename])