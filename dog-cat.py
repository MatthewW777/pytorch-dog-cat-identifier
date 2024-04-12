import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(90),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load your dataset
dataset = ImageFolder(root="-", transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Adjust the input size of fc1 according to the output size of the last conv layer
        self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 13 * 13)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # Learning rate and momentum
for epoch in range(10): # Number of epochs
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999: # Print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 2000}')
            running_loss = 0.0
torch.save(model.state_dict(), 'model.pth')

print('Finished Training')

# Evaluate the model's performance on the validation set
correct = 0
total = 0
with torch.no_grad():  # Disable gradient computation
    for data in val_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the model on the validation images: {accuracy:.2f}%')
model = SimpleCNN()
model.load_state_dict(torch.load('-'))
model.eval()  # Set the model to evaluation mode


def process_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Same as during training
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Same normalization as training
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add a batch dimension
    return image
def predict_image(image_path, model):
    # Process the image
    image = process_image(image_path)
    
    # Make a prediction
    output = model(image)
    _, predicted = torch.max(output, 1)
    
    # Convert the prediction to a class name
    classes = ['cat', 'dog']  # Assuming 0 = cat, 1 = dog
    prediction = classes[predicted[0]]
    
    return prediction
image_path = "-`.jpg"
prediction = predict_image(image_path, model)
print(f"This image is a {prediction}.")
