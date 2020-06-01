## Part 1 Basic ML

##lab05_logistic_regression



```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
```

```python
x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]

y_data =[[0],[0],[0],[1],[1],[1]]
```

```python
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)
```

```python
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)
```

```python
print(x_train.shape)
print(y_train.shape)
```

```python
print('e^1 equals:', torch.exp(torch.FloatTensor([1])))
```

```python
W = torch.zeros((2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```

```python
hypothesis = 1/(1+torch.exp(-(x_train.matmul(W)+b)))
```



```python
#Whole Training Procedure
#모델 초기화
W = torch.zeros((2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W,b], lr=1)

nb_epochs=1000
for epoch in range(nb_epochs+1):

  hypothesis = torch.sigmoid(x_train.matmul(W)+b)
  cost = F.binary_cross_entropy(hypothesis, y_train)

  optimizer.zero_grad()
  cost.backward()
  optimizer.step()

  if epoch %100 ==0:
    print(epoch, nb_epochs, cost.item())
```

