import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
import csv
import dvc.api
import os
import mlflow
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.preprocessing.image import load_img, img_to_array

mlflow.set_tracking_uri("http://127.0.0.1:5000")
#mlflow.set_tracking_uri("file:///C:/Users/boyan/PycharmProjects/pythonProject1")
mlflow.set_experiment("mlflow_MNIST_classifier_experiment2")

with mlflow.start_run(run_name="MNIST_classificator") as run:

# params = dvc.api.params_show()

    pathtodata = 'data/combined_data.pt'
    def create_nn(batch_size=200, learning_rate=0.01, epochs=10,
                  log_interval=10):
        with mlflow.start_run(run_name="data", nested=True) as data:

            # Загрузка объединенных данных
            with open(pathtodata, 'rb') as f:
                combined_data = torch.load(f)
            mlflow.log_param("path to data", pathtodata)

            # Создание экземпляра TensorDataset с использованием объединенных данных
            train_dataset = combined_data
            mlflow.log_artifact("C:/Users/boyan/PycharmProjects/pythonProject1/data/combined_data.pt")

            # Создание загрузчика данных
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            test_loader = torch.utils.data.DataLoader(
                datasets.MNIST('../data', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])),
                batch_size=batch_size, shuffle=True)

            class Net(nn.Module):
                def __init__(self):
                    super(Net, self).__init__()
                    self.fc1 = nn.Linear(28 * 28, 200)
                    self.fc2 = nn.Linear(200, 200)
                    self.fc3 = nn.Linear(200, 10)
                def forward(self, x):
                    x = F.relu(self.fc1(x))
                    x = F.relu(self.fc2(x))
                    x = self.fc3(x)
                    return F.log_softmax(x)

            net = Net()
            print(net)

            # create a stochastic gradient descent optimizer
            optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
            # create a loss function
            criterion = nn.NLLLoss()

        with mlflow.start_run(run_name="training", nested=True) as training:
            # run the main training loop
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch size", batch_size)
            mlflow.log_param("learning rate", learning_rate)
            mlflow.log_param("log interval", log_interval)

            for epoch in range(epochs):
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = Variable(data), Variable(target)
                    # resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
                    data = data.view(-1, 28*28)
                    optimizer.zero_grad()
                    net_out = net(data)
                    loss = criterion(net_out, target)
                    loss.backward()
                    optimizer.step()
                    if batch_idx % log_interval == 0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, batch_idx * len(data), len(train_loader.dataset),
                                   100. * batch_idx / len(train_loader), loss.item()))

                    mlflow.log_metric("loss", loss.item(), step=batch_idx)
    
            #torch.save(net, 'data/model.pth')

        with mlflow.start_run(run_name="testing", nested=True) as testing:

            # run a test loop
            test_loss = 0
            correct = 0
            for data, target in test_loader:
                data, target = Variable(data, volatile=True), Variable(target)
                data = data.view(-1, 28 * 28)
                net_out = net(data)
                # sum up batch loss
                test_loss += criterion(net_out, target).item()
                pred = net_out.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data).sum()
                accuracy = correct / len(test_loader.dataset)
                loss_test = test_loss / len(test_loader.dataset)
                mlflow.log_metric("loss", loss_test, step=batch_idx)
                mlflow.log_metric("accuracy", accuracy.item(), step=batch_idx)

        with mlflow.start_run(run_name="model", nested=True) as model:
            test_loss /= len(test_loader.dataset)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
            # вказуємо шлях до файлу, в який будемо зберігати Accuracy

            file_path = os.path.join(os.getcwd(), 'accuracy.txt')

            # відкриваємо файл для запису
            with open(file_path, 'w') as f:
                # записуємо Accuracy у файл
                f.write('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                    test_loss, correct, len(test_loader.dataset),
                    100. * correct / len(test_loader.dataset)))
            mlflow.log_artifact("C:/Users/boyan/PycharmProjects/pythonProject1/accuracy.txt")

            images, labels = next(iter(train_loader))
            a = 0
            plt.imshow(images[a].reshape(28, 28), cmap="gray")
            print("Predicted: ", net(images[a].view(-1, 28 * 28)), "True: ", labels[a])
            pathtoimg = 'C:/Users/boyan/PycharmProjects/pythonProject1/image.jpg'
            plt.savefig(pathtoimg)
            plt.show()
            # Сохраняем изображение в файл
            # Загружаем изображение в MLflow
            img = load_img(pathtoimg, target_size=(224, 224))
            img_filename = os.path.basename(pathtoimg)

            new_image = Image.new('RGB', (img.width, img.height + 30), color='white')
            new_image.paste(img, (0, 30))
            # добавить заголовок
            draw = ImageDraw.Draw(new_image)
            font = ImageFont.truetype('arial.ttf', size=20)
            text = f"Class: {labels[a]}"
            text_width, text_height = draw.textsize(text, font)
            x = (new_image.width - text_width) // 2
            y = 0
            draw.text((x, y), text, font=font, fill='black')

            # сохранить изображение
            class_dir = os.path.join("C:/Users/boyan/PycharmProjects/pythonProject1")
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            output_file = os.path.join(class_dir, f"mlflow_{img_filename}")
            # output_file = os.path.join(class_dir)
            new_image.save(output_file)

            mlflow.log_artifact("mlflow_image.jpg")


    create_nn()