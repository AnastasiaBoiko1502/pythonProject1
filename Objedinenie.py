import torch
import csv
import os

# Считывание названий файлов из файла csv
filenames = []
with open(os.path.join('data', 'data_filenames.csv'), 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Пропустить заголовок
    for row in reader:
        filenames.append(row[0])

# Загрузка данных из файлов и объединение их
combined_data = torch.load('data/train_chunk_0.pt')
for i, filename in enumerate(filenames):
    with open(os.path.join('data', filename), 'rb') as f:
        chunk = torch.load(f)
        if i == 0:
            combined_data = chunk
        else:
            combined_data = torch.utils.data.ConcatDataset([combined_data, chunk])

# Сохранение объединенных данных в файл
with open(os.path.join('data', 'combined_data.pt'), 'wb') as f:
    torch.save(combined_data, f)

# Вывод количества элементов в объединенных данных
print(f'Number of elements in combined data: {len(combined_data)}')
