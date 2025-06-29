import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 定义第一个卷积层，输入通道数为1，输出通道数为32，卷积核大小为3，步长为1
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1)
        # 定义第二个卷积层，输入通道数为32，输出通道数为64，卷积核大小为3，步长为1
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1)
        # 定义dropout层，dropout率为0.5
        self.dropout = torch.nn.Dropout(0.5)
        # 定义第一个全连接层，输入特征数为1600，输出特征数为128
        self.fc1 = torch.nn.Linear(1600, 128)  # 1600 = 64 * 5 * 5
        # 定义第二个全连接层，输入特征数为128，输出特征数为10
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        # 对输入进行第一个卷积操作，并使用ReLU激活函数
        x = torch.relu(self.conv1(x))      # [N, 32, 26, 26]
        # 对第一个卷积操作的结果进行最大池化操作
        x = torch.max_pool2d(x, 2)         # [N, 32, 13, 13]
        # 对最大池化操作的结果进行第二个卷积操作，并使用ReLU激活函数
        x = torch.relu(self.conv2(x))      # [N, 64, 11, 11]
        # 对第二个卷积操作的结果进行最大池化操作
        x = torch.max_pool2d(x, 2)         # [N, 64, 5, 5]
        # 将最大池化操作的结果进行展平操作
        x = torch.flatten(x, 1)           # [N, 1600]
        # 对展平操作的结果进行dropout操作
        x = self.dropout(x)
        # 对dropout操作的结果进行第一个全连接操作，并使用ReLU激活函数
        x = torch.relu(self.fc1(x))
        # 对第一个全连接操作的结果进行第二个全连接操作，并使用log_softmax激活函数
        x = torch.log_softmax(self.fc2(x), dim=1)
        # 返回结果
        return x

def get_data_loader(is_train):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = MNIST(root='mnist_data/', train=is_train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True)

def evaluate(test_data, net):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_data:
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
            total += y.size(0)
    return correct / total

def main():
    train_loader = get_data_loader(is_train=True)
    test_loader = get_data_loader(is_train=False)
    model = CNN().to(device)
    
    print("Initial accuracy:", evaluate(test_loader, model))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = torch.nn.functional.nll_loss(outputs, y)
            loss.backward()
            optimizer.step()
        acc = evaluate(test_loader, model)
        print(f"Epoch {epoch+1}: Accuracy {acc:.4f}")

    torch.save(model.state_dict(), 'cnn_model.pth')
    print("CNN model saved to cnn_model.pth")

    # Visualization
    test_images, _ = next(iter(test_loader))
    test_images = test_images.to(device)[:4]
    outputs = model(test_images)
    preds = outputs.argmax(dim=1)

    plt.figure(figsize=(10, 5))
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.imshow(test_images[i].cpu().squeeze(), cmap='gray')
        plt.title(f"Pred: {preds[i].item()}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
