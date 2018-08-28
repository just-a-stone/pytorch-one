import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms


class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()

        self.l1 = torch.nn.Linear(784, 520)
        self.l2 = torch.nn.Linear(520, 320)
        # self.l1 = torch.nn.Linear(784, 600)
        # self.l2 = torch.nn.Linear(600, 450)
        # self.ll = torch.nn.Linear(450, 320)
        self.l3 = torch.nn.Linear(320, 240)
        self.l4 = torch.nn.Linear(240, 120)
        self.l5 = torch.nn.Linear(120, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        # x = F.relu(self.ll(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))

        # return F.log_softmax(x)
        return x

model = Model()


# criterion = torch.nn.NLLLoss()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# load data
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('~/tmp/ml/data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])),
    batch_size=64, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('~/tmp/ml/data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])),
    batch_size=100, shuffle=True
)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print('epoch: {} [{}/{} ({:.0f})]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]
            ))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data), Variable(target)
        output = model(data)

        test_loss += criterion(output, target).data[0]

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('test set: average loss: {:.4f}, accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))

for epoch in range(1, 10):
    train(epoch)
    test()