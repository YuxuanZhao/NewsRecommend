from torch import no_grad
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.nn import Module, Conv2d, MaxPool2d, ReLU, Sequential, Flatten, Linear, Dropout, CrossEntropyLoss
from torch.nn.init import xavier_uniform_
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam

# lr = 5e-2 # SGD
lr = 1e-3 # Adam
batch_size = 64
num_epochs = 5
loss_function = CrossEntropyLoss()

class Block(Module):
    def __init__(self, num_convs, in_channels, out_channels):
        super().__init__()
        layers = []
        layers.append(Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))
        layers.append(ReLU())
        for _ in range(num_convs - 1):
            layers.append(Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(ReLU())
        layers.append(MaxPool2d(kernel_size=2, stride=2))
        self.network = Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
class VGG(Module):
    def __init__(self):
        super().__init__()
        self.network = Sequential(
            Block(1, 3, 64),
            Block(2, 64, 128),
            Flatten(),
            Linear(128 * 64, 4096),
            ReLU(),
            Dropout(0.5),
            Linear(4096, 10)
        )
        self.network.apply(self.init_weights)
        self.network.to(device)

    def forward(self, x):
        return self.network(x)

    def init_weights(self, model):
        if isinstance(model, (Linear, Conv2d)):
            xavier_uniform_(model.weight)
    
def accuracy(outputs, targets):
    if outputs.dim() > 1 and outputs.size(1) > 1:
        outputs = outputs.argmax(dim=1)
    return (outputs == targets).sum().item()

train_dataset = CIFAR10('./CIFAR10', train=True, transform=ToTensor(), download=True)
test_dataset = CIFAR10('./CIFAR10', train=False, transform=ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
device = 'cuda'
writer = SummaryWriter('logs')
vgg = VGG()
optimizer = Adam(vgg.parameters(), lr=lr)
num_batches = len(train_loader)

for epoch in range(num_epochs):
    vgg.train()
    train_loss, running_correct = 0, 0
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = vgg(inputs)
        batch_loss = loss_function(outputs, targets)
        batch_loss.backward()
        optimizer.step()
        
        train_loss += batch_loss.item()
        running_correct += (outputs.argmax(1) == targets).sum()

    train_acc = running_correct / len(train_dataset)

    vgg.eval()
    test_loss, test_correct = 0, 0
    with no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = vgg(inputs)
            test_loss += loss_function(outputs, targets).item()
            test_correct += (outputs.argmax(1) == targets).sum()
    test_acc = test_correct / len(test_dataset)

    writer.add_scalars("VGG/Accuracy", {"Train": train_acc, "Test": test_acc}, global_step=epoch)
    writer.add_scalars("VGG/Loss", {"Train": train_loss, "Test": test_loss}, global_step=epoch)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

writer.close()
# tensorboard --logdir=logs --port=6007