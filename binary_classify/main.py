import torch
import numpy as np


class SimpleNN(torch.nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.l1 = torch.nn.Linear(in_size, hid_size)
        self.l2 = torch.nn.Linear(hid_size, out_size)

    def forward(self, x):
        h = torch.relu(self.l1(x))
        o = torch.sigmoid(self.l2(h))
        return o


def target_func(x):
    y = x[:, 0] * x[:, 0] + x[:, 1] * 2.0
    ans = torch.where(y < 0.0, torch.tensor(0.0, dtype=torch.float32), torch.tensor(1.0, dtype=torch.float32))
    return ans.reshape([-1, 1])


model = SimpleNN(2, 3, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

for epoch in range(5000):
    rnd = [[np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)] for _ in range(100)]
    x = torch.tensor(rnd)

    y = model(x)
    y_hat = target_func(x)
    E = torch.nn.functional.binary_cross_entropy(y, y_hat, reduction="sum")

    optimizer.zero_grad()
    E.backward()
    optimizer.step()


rnd = [[np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)] for _ in range(10)]
x = torch.tensor(rnd)
y = model(x)
y_hat = target_func(x)

result = list(zip(y.data, y_hat.data))
for yy, yh in result:
    print(yy, yh)
