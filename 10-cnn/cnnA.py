import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms


class InceptionA(torch.nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()

        self.branch1x1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch5x5_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = torch.nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3db1_2 = torch.nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3db1_3 = torch.nn.Conv2d(24, 24, kernel_size=3, padding=1)

        self.branch_pool = torch.nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_2(self.branch5x5_1(x))

        branch3x3 = self.branch3x3db1_3(self.branch3x3db1_2(self.branch3x3dbl_1(x)))

        branch_pool = self.branch_pool(F.max_pool2d(x, kernel_size=3, stride=1, padding=1))

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]

        return torch.cat(outputs, 1)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(88, 20, kernel_size=5)

        self.incep1 = InceptionA(10)
        self.incep2 = InceptionA(20)

        self.mp = torch.nn.MaxPool2d(kernel_size=2)
        self.fc = torch.nn.Linear(1408, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return F.log_softmax(x)

model = Model()

criterion = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# load data
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('~/tmp/ml/data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])),
    batch_size=100, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('~/tmp/ml/data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])),
    batch_size=100, shuffle=True
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print('epoch {} {}/{} ({:.0f}%)\tLoss:{}'.format(
                epoch, batch_idx, len(train_loader),
                100. * batch_idx / len(train_loader), loss
            ))

def test(epoch):
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('epoch {} average loss {:.6f}. correct: {}/{} ({:.2f}%)'.format(
        epoch, test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)
    ))


for epoch in range(1, 6):
    train(epoch)
    test(epoch)