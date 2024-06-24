import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
import torch
import os
import torch.nn.functional as F
from PIL import Image
import seaborn as sns
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torchvision

def contarCorrectas(net, batch, labels, func=None):
    '''Dado un batch y sus etiquetas, cuenta el número de respuestas
    correctas de una red. El parámetro func aplica una modificación al
    tensor que contiene los datos.'''

    if func is not None:
        batch = func(batch)
    salidas = net(batch)
    respuestas = salidas.max(dim=1)[1]
    cantidadCorrectas = (respuestas == labels).sum()
    return cantidadCorrectas.item()

def calcularPrecisionGlobal(net, data_loader, batch_size, func=None, cuda=False):
    '''Calcula la precisión de una red dado un data_loader.
    Recibe una función que transforma los datos en caso de ser necesario.'''

    correctas = 0
    for images, labels in data_loader:
        if cuda and torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        correctas += contarCorrectas(net, images, labels, func)

    return (100 * correctas) / (len(data_loader) * batch_size)

# Definimos las clases
classes = ('pajaro', 'perro', 'rana')

# Definimos transformaciones de aumento de datos
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor()
])

# Función para cargar imágenes y etiquetas
def cargar_imagenes(ruta, transform):
    imagenes = []
    etiquetas = []
    for etiqueta, clase in enumerate(classes):
        ruta_clase = os.path.join(ruta, clase)
        for archivo in os.listdir(ruta_clase):
            if archivo.endswith(".jpeg") or archivo.endswith(".jpg") or archivo.endswith(".png"):
                img_path = os.path.join(ruta_clase, archivo)
                img = Image.open(img_path).convert('RGB')
                img = transform(img)
                imagenes.append(img)
                etiquetas.append(etiqueta)
    return imagenes, etiquetas

train_images, train_labels = cargar_imagenes("Train", transform)
train_data = TensorDataset(torch.stack(train_images), torch.tensor(train_labels))
train_loader = DataLoader(train_data, batch_size=4, shuffle=True)

#Queremos saber las medias de la imagen
train_features, train_labels = next(iter(train_loader))
train_features.shape

class Net(nn.Module):
    def __init__(self):
        '''
        Construcción de la Red, define las capas que se utiizaran.
        '''
        super(Net, self).__init__()

        # Toma en cuenta el constructor de las capas convolucionales
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        '''
        Define el orden con el que se realizará la propagación hacia adelante
        de la red.
        '''
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def custom_train(self, epochs, data_loader, criterion, optimizer, cuda=False):
        '''
        Define una función de entrenamiento, teniendo en cuenta la forma en la que llegan
        los datos, e iteramos sobre estos.
        '''

        loss_values = []

        if cuda and torch.cuda.is_available():
            self.cuda()

        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(data_loader, 0):
                inputs, labels = data
                if cuda and torch.cuda.is_available():
                    inputs, labels = inputs.cuda(), labels.cuda()

                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / len(data_loader)
            loss_values.append(epoch_loss)
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}')

        plt.plot(loss_values)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()

# Instanciamos la red
net = Net()

# Definimos la función de pérdida y el optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Entrenamos la red
net.custom_train(10, train_loader, criterion, optimizer, cuda=True)

# Calculamos la precisión global
prec_train = calcularPrecisionGlobal(net, train_loader, 4, cuda=True)
print("Precisión Global: %.4f%%"%(prec_train))

"""# VISUALIZACION RESULTADOS DEL MODELO"""

def test(model, x):
    '''
    Función que recibe un modelo de la carpeta test y prueba cada imagen
    '''
    model.eval()
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    img = Image.open(x).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)
    output = model(img)
    _, predicted = torch.max(output, 1)
    return classes[predicted.item()]

test_image_paths = ['Test/test1.jpg', 'Test/test2.jpeg', 'Test/descarga.jpeg', 'Test/rana.jpeg']
for image_path in test_image_paths:
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title(f'Clase: {test(net, image_path)}')
    plt.show()

def get_matrix(model, data_loader, classes, name):
    '''Matriz de confusión en el que se muestra la cantidad de predicciones correctas'''
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, labels in data_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.view(-1).cpu().numpy())
            all_labels.extend(labels.view(-1).cpu().numpy())
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title("Confusion Matrix on {typeSet}".format(typeSet = name))
    plt.show()

get_matrix(net, train_loader, classes, "train")

def plot_confusion_matrix(model, data_loader, classes):
    model.eval()
    y_true = []
    y_pred = []
    for images, labels in data_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true += labels.cpu().numpy().tolist()
        y_pred += predicted.cpu().numpy().tolist()
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix on test')
    plt.show()

plot_confusion_matrix(net, train_loader, classes)

"""# NORMALIZACION DE IMAGENES"""

class Net(nn.Module):
    def __init__(self):
        '''
        Construcción de la Red, define las capas que se utiizaran.
        '''
        super(Net, self).__init__()

        # Toma en cuenta el constructor de las capas convolucionales
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        '''
        Define el orden con el que se realizará la propagación hacia adelante
        de la red.
        '''
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def custom_train(self, epochs, data_loader, criterion, optimizer, cuda=False):
        '''
        Define una función de entrenamiento, teniendo en cuenta la forma en la que llegan
        los datos, e iteramos sobre estos.
        '''

        loss_values = []

        if cuda and torch.cuda.is_available():
            self.cuda()

        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(data_loader, 0):
                inputs, labels = data
                if cuda and torch.cuda.is_available():
                    inputs, labels = inputs.cuda(), labels.cuda()

                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / len(data_loader)
            loss_values.append(epoch_loss)
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}')

        plt.plot(loss_values)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()

red=Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(red.parameters(), lr=0.001)
red.custom_train(10, train_loader, criterion, optimizer, cuda=True)

pred_train = calcularPrecisionGlobal(net, train_loader, 4)
print("Precisión Global: %.4f%%"%(pred_train))

get_matrix(red, train_loader, classes, "train")
get_matrix(red, train_loader, classes, "test")