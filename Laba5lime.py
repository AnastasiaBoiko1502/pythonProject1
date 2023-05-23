import os
import torch
import numpy as np
from PIL import Image
from lime import lime_image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
from skimage.segmentation import mark_boundaries
import wandb

wandb.login(key="f22539d7913dafa5774ec2b12eb6b20057ea3b07")
wandb.init(project="MNIST classificator W&B", entity="boyanastya", name="lime interpretator run")

CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
directory = 'visualized'

if not os.path.exists(directory):
    os.makedirs(directory)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def get_pil_transform():
    transf = transforms.Compose([
        transforms.Resize((28, 28))
    ])
    return transf


def get_preprocess_transform():
    transf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    return transf


def batch_predict(images):
    model.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)

    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()


model = Net()
model.load_state_dict(torch.load("model.pth"))
images = {
    image_name: Image.open(os.path.join("digits", image_name)).resize((224, 224))
    for image_name in os.listdir("digits")
}

for name, img in images.items():
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        img = img.convert('RGB')

    pil_transform = get_pil_transform()
    preprocess_transform = get_preprocess_transform()

    test_pred = batch_predict([pil_transform(img)])
    result = test_pred.squeeze().argmax()
    print(result)

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(np.array(pil_transform(img)), batch_predict, top_labels=5, hide_color=0, num_samples=5)

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=1, hide_rest=False)
    img_boundary = mark_boundaries(temp / 255.0, mask)

    class_path = os.path.join(directory, str(result))
    if not os.path.exists(class_path):
        os.makedirs(class_path)

    plt.imshow(img_boundary)
    plt.savefig(os.path.join(class_path, f"visualized_{name}"))
    plt.close()

wandb.log_artifact("C:/Users/boyan/PycharmProjects/pythonProject1/visualized")


wandb.finish()