import torch
import csv

# Считывание названий файлов из файла csv
filenames = []
with open('data_filenames.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Пропустить заголовок
    for row in reader:
        filenames.append(row[0])

# Загрузка данных из файлов и объединение их
combined_data = torch.load('data/train_chunk_0.pt')
for filename in filenames:
    with open(filename, 'rb') as f:
        chunk = torch.load(f)
        combined_data = torch.cat((combined_data, chunk)) # объединяем данные с помощью функции torch.cat()

# Сохранение объединенных данных в файл
with open('data/combined_data.pt', 'wb') as f:
    torch.save(combined_data, f)

# Вывод количества элементов в объединенных данных
print(f'Number of elements in combined data: {len(combined_data)}')