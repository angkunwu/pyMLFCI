import torch
import math

dtype = torch.float
device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.set_default_device(device)

x = torch.linspace(-math.pi,math.pi,2000,dtype=dtype)
y = torch.sin(x)

a = torch.randn((),dtype=dtype,requires_grad=True)
b = torch.randn((),dtype=dtype,requires_grad=True)
c = torch.randn((),dtype=dtype,requires_grad=True)
d = torch.randn((),dtype=dtype,requires_grad=True)

learning_rate = 1e-6
for t in range(2000):
    y_pred = a + b*x + c*x**2 + d*x**3
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())
    loss.backward()
    with  torch.no_grad(): #Manually update weights using gradient descent. Wrap in torch.no_grad()
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')

class LegendrePolynomial3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return 0.5 * (5*input**3 - 3*input)
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * 1.5 *(5*input**2 - 1)
# 4 weights: y = a + b * P3(c + d * x), these weights need to be initialized
a = torch.full((),0.0,dtype=dtype,requires_grad=True)
b = torch.full((),-1.0,dtype=dtype,requires_grad=True)
c = torch.full((),0.0,dtype=dtype,requires_grad=True)
d = torch.full((),0.3,dtype=dtype,requires_grad=True)
for t in range(2000):
    P3 = LegendrePolynomial3.apply
    y_pred = a + b*P3(c + d*x) # forward pass
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())
    loss.backward()
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

print(f'Result: y = {a.item()} + {b.item()} * P3({c.item()} + {d.item()} x)')


y_pred = a + b*x + c*x**2 + d*x**3

import matplotlib.pyplot as plt

# plot the comparison
plt.figure()
plt.plot(x.cpu().numpy(),y.cpu().numpy(),label='y')
plt.plot(x.cpu().numpy(),y_pred.cpu().numpy(),label='y_pred')
plt.show()


# y is a linear function of (x,x^2,x^3)
p = torch.tensor([1,2,3])
xx = x.unsqueeze(-1).pow(p) # x.unsqueeze(-1) has shape (2000,1), xx has shape (2000,3)
model = torch.nn.Sequential(
    torch.nn.Linear(3,1), #computes output from input using linear function
    # and holds internal tensors for its weights and bias
    torch.nn.Flatten(0,1) # flatens the output of the linear layer to a 1d tensor to match y
)
loss_fn = torch.nn.MSELoss(reduction='sum') # Mean Squared Error loss

for t in range(2000):
    y_pred = model(xx)
    loss = loss_fn(y_pred,y)
    if t % 100 == 99:
        print(t,loss.item())
    model.zero_grad() # zero the gradients before running the backward pass
    loss.backward() # compute gradient of the loss for all parameters
    with torch.no_grad(): # update the weights using gradient descent
        for param in model.parameters():
            param -= learning_rate * param.grad

linear_layer = model[0]

print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:,0].item()} x + {linear_layer.weight[:,1].item()} x^2 + {linear_layer.weight[:,2].item()} x^3')

# the optim package abstracts the idea of an optimization algorithm and provides implementations of commonly used optimization algorithms
# optimization beyond stochastic gradient descent
optimizer = torch.optim.RMSprop(model.parameters(),lr=learning_rate)
for t in range(2000):
    y_pred = model(xx)
    loss = loss_fn(y_pred,y)
    if t % 100 == 99:
        print(t,loss.item())
    model.zero_grad() # zero the gradients before running the backward pass
    loss.backward() # compute gradient of the loss for all parameters
    optimizer.step()



# subclassing nn.Module
class Polynomial3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))
    def forward(self,x):
        return self.a + self.b*x + self.c*x**2 + self.d*x**3
    def string(self):
        return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3'


model = Polynomial3()
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(),lr=1e-6)
for t in range(2000):
    y_pred = model(x)
    loss = criterion(y_pred,y)
    if t % 100 == 99:
        print(t,loss.item())
    optimizer.zero_grad() # zero gradients
    loss.backward()
    optimizer.step()
print(f'Result: {model.string()}')


# dynamic graphs and weight sharing: a third-fifth order polynomial,
# that on each forward pass chooses a random number between 3 and 5
# and uses that many orders, reusing the smae weights multiple times
# to compute the 4th and 5th order
import random
class DynamicNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))
        self.e = torch.nn.Parameter(torch.randn(()))
    def forward(self,x):
        y = self.a + self.b*x + self.c*x**2 + self.d*x**3
        for exp in range(4, random.randint(4,6)):
            y = y + self.e * x**exp
        return y
    def string(self):
        return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3 + {self.e.item()} x^4 ? + {self.e.item()} x^5 ?'

model = DynamicNet()
optimizer = torch.optim.SGD(model.parameters(),lr=1e-8,momentum=0.9)
# vanilla SGD is tough, we use momentum to stabilize the learning
for t in range(30000):
    y_pred = model(x)
    loss = criterion(y_pred,y)
    if t % 2000 == 1999:
        print(t,loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(f'Result: {model.string()}')

