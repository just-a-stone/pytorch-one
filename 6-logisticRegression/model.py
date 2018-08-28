import torch
from torch.autograd import Variable

x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0], [4.0]]))
y_data = Variable(torch.Tensor(([0.], [0.], [1.], [1.])))

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))


model = Model()



# loss & optim
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# training
for epoch in range(500):
    y_pred = model(x_data)

    loss = criterion(y_pred, y_data)
    print('epoch %d loss %.6f' % (epoch, loss))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# after training
hour_var = Variable(torch.Tensor([[0.5]]))
print('predict (after training)', 0.5, model.forward(hour_var).data[0][0])
hour_var = Variable(torch.Tensor([[7.0]]))
print('predict (after training)', 7.0, model.forward(hour_var).data[0][0])