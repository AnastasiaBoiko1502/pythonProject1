import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import wandb
import numpy as np
from lime import lime_image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Sequential
from skimage.segmentation import mark_boundaries
from tensorflow.keras.layers import Dense

wandb.login(key="f22539d7913dafa5774ec2b12eb6b20057ea3b07")
wandb.init(project="MNIST classificator W&B", entity="boyanastya", name="classification run")

pathtodata = 'data/combined_data.pt'


def create_nn(batch_size=200, learning_rate=0.01, epochs=10,log_interval=10):
    # Загрузка объединенных данных
    with open(pathtodata, 'rb') as f:
        combined_data = torch.load(f)
    wandb.log({"path to data": pathtodata})

    # Создание экземпляра TensorDataset с использованием объединенных данных
    train_dataset = combined_data
    wandb.log_artifact("C:/Users/boyan/PycharmProjects/pythonProject1/data/combined_data.pt")

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


    # run the main training loop
    wandb.log({"epochs": epochs,
               "batch size": batch_size,
               "learning rate": learning_rate,
               "log interval": log_interval})

    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)
            # resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
            data = data.view(-1, 28 * 28)
            optimizer.zero_grad()
            net_out = net(data)
            loss = criterion(net_out, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                            100. * batch_idx / len(train_loader), loss.item()))

            wandb.log({"loss": loss.item()}, step=batch_idx)

    #torch.save(net, 'data/model.pth')
    #model_path = 'C:/Users/boyan/PycharmProjects/pythonProject1/model.h5'
    #torch.save(net.state_dict(), model_path)
    #wandb.log_artifact(model_path)

    # Создание экземпляра модели Sequential из Keras
    model = Sequential()
    model.add(Dense(200, activation='relu', input_shape=(28 * 28,)))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    # Копирование весов из PyTorch модели в Keras модель
    model.layers[0].set_weights([net.fc1.weight.t().detach().numpy(), net.fc1.bias.detach().numpy()])
    model.layers[1].set_weights([net.fc2.weight.t().detach().numpy(), net.fc2.bias.detach().numpy()])
    model.layers[2].set_weights([net.fc3.weight.t().detach().numpy(), net.fc3.bias.detach().numpy()])
    # Сохранение модели в формате .h5
    model_path = 'C:/Users/boyan/PycharmProjects/pythonProject1/model.h5'
    model.save(model_path)
    #wandb.log_artifact(model_path)
    wandb.log_artifact("C:/Users/boyan/PycharmProjects/pythonProject1/model.pth")

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
        accuracy = correct
        loss_test = test_loss
        wandb.log({"loss": loss_test, "accuracy": accuracy.item()}, step=batch_idx)


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
    wandb.log_artifact("C:/Users/boyan/PycharmProjects/pythonProject1/accuracy.txt")


    images, labels = next(iter(train_loader))
    for i in range(10):
        plt.imshow(images[i].reshape(28, 28), cmap="gray")
        print("Predicted:", net(images[i].view(-1, 28 * 28)), "True:", labels[i])
        plt.savefig(f"C:/Users/boyan/PycharmProjects/pythonProject1/image_{i}.jpg")
        plt.close()

    image_dir = os.path.join("C:/Users/boyan/PycharmProjects/pythonProject1")
    for i in range(10):
        img_path = os.path.join(image_dir, f"image_{i}.jpg")
        img = load_img(img_path, target_size=(224, 224))
        img_filename = os.path.basename(img_path)
        new_image = Image.new('RGB', (img.width, img.height + 30), color='white')
        new_image.paste(img, (0, 30))
        draw = ImageDraw.Draw(new_image)
        font = ImageFont.truetype('arial.ttf', size=20)
        text = f"Predicted class: {labels[i]}"
        text_width, text_height = draw.textsize(text, font)
        x = (new_image.width - text_width) // 2
        y = 0
        draw.text((x, y), text, font=font, fill='black')

        output_file = os.path.join(image_dir, f"wandb_{img_filename}")
        new_image.save(output_file)
        class_dir = os.path.join("C:/Users/boyan/PycharmProjects/pythonProject1/image_result", str(labels[i]))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        output_file = os.path.join(class_dir, f"wandb_{img_filename}")
        new_image.save(output_file)

        #wandb.log_artifact("C:/Users/boyan/PycharmProjects/pythonProject1/image_result")


create_nn()
wandb.log_artifact("C:/Users/boyan/PycharmProjects/pythonProject1/image_result")
wandb.finish()



"""wandb.init(project="MNIST classificator W&B", entity="boyanastya", name="lime interpretator run")

class_names = os.listdir('image_result')

def prepare_image_for_model(image):
    image_array = np.array(image)
    resized_image = np.resize(image_array, 784)
    return resized_image / 255.0

def create_image_explanations(images, model):
    explainer = lime_image.LimeImageExplainer()
    for name, image in images.items():
        preprocessed_image = prepare_image_for_model(image).astype('double')
        predictions = model.predict(np.expand_dims(preprocessed_image, axis=0))
        explanation = explainer.explain_instance(preprocessed_image, model.predict, num_samples=250)
        visualize_explanation(explanation, name, predictions)


def visualize_explanation(explanation, name, predictions):
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5,
                                                hide_rest=False)
    plt.imshow(mark_boundaries(temp, mask))
    class_idx = np.argmax(predictions[0])
    class_name = class_names[class_idx]

    class_path = os.path.join(directory, class_name)
    if not os.path.exists(class_path):
        os.makedirs(class_path)

    plt.savefig(os.path.join(class_path, f"lime_{name}"))
    plt.close()


directory = 'lime_interpretator'
if not os.path.exists(directory):
    os.makedirs(directory)


model = load_model("C:/Users/boyan/PycharmProjects/pythonProject1/model.h5")
images = {image_name: Image.open(os.path.join("C:/Users/boyan/PycharmProjects/pythonProject1/images", image_name)).resize((28, 28)) for image_name in os.listdir("C:/Users/boyan/PycharmProjects/pythonProject1/images")}
create_image_explanations(images, model)

wandb.log_artifact("C:/Users/boyan/PycharmProjects/pythonProject1/lime_interpretator")


wandb.finish()"""