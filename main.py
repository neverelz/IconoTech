import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt


df = pd.read_csv("icons_dataset_3.csv", sep=";")
df["image_path"] = df["id"].apply(lambda x: os.path.join("JPGs", f"{x}.jpg"))
df = df[df["image_path"].apply(os.path.exists)]


# энкодинг меток
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["Тип изображения"].astype(str))
# удаление классов с <2 obj
label_counts = df["label"].value_counts()
valid_labels = label_counts[label_counts >= 2].index
df = df[df["label"].isin(valid_labels)]
# снова энкодинг меток
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["Тип изображения"].astype(str))


train_df, test_df = train_test_split(df, test_size=0.5, stratify=df["label"], random_state=42)


# работа с датасетом (разобраться)
class IconDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = Image.open(self.df.loc[idx, "image_path"]).convert("RGB")
        label = self.df.loc[idx, "label"]
        if self.transform:
            image = self.transform(image)
        return image, label


# подготовка данных (трансфорсмация в тензор)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_dataset = IconDataset(train_df, transform=transform)
test_dataset = IconDataset(test_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)


# структура нейронки
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("GPU available:", torch.cuda.is_available())

num_classes = df["label"].nunique()
model = SimpleCNN(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# обучение модельки
for epoch in range(5):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


# тестирование
model.eval()
correct, total = 0, 0 # чзх, разобраться

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

model.eval()
class_names = label_encoder.classes_

# выбираем случайные 5 изображений из тестового набора
samples = random.sample(range(len(test_dataset)), 5)

plt.figure(figsize=(15, 5))
for i, idx in enumerate(samples):
    image, true_label = test_dataset[idx]
    input_image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_image)
        _, predicted_label = torch.max(output, 1)

    true_name = class_names[true_label]
    predicted_name = class_names[predicted_label.item()]

    image = image.permute(1, 2, 0).cpu().numpy()  # преобразуем для plt.imshow

    plt.subplot(1, 5, i + 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Истинно: {true_name}\nПредсказано: {predicted_name}", fontsize=9)

plt.tight_layout()
plt.show()


# добавить вывод предиктов