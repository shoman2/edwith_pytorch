## Part 3 - CNN

lab-10_3_Visdom

Visdom = 파이토치로 구현하는 걸 시각화해서 볼 수 있음



```python
import torch
import torch.nn as nn

import torchvision
import torchvision.datasets as dsets

##

import visdom
vis = visdom.Visdom()

```



### 텍스트

```python
vis.text("Hello, world!",env="main")

```

### 이미지

```python
a=torch.randn(3,200,200)
vis.image(a)

```

### MNIST & CIFAR10

```python
MNIST = dsets.MNIST(root="./MNIST_data",train = True,transform=torchvision.transforms.ToTensor(), download=True)
cifar10 = dsets.CIFAR10(root="./cifar10",train = True, transform=torchvision.transforms.ToTensor(),download=True)

```

#### CIFAR10

```python
data = cifar10.__getitem__(0)
print(data[0].shape)
vis.images(data[0],env="main")

```

#### MNIST

```python
data = MNIST.__getitem__(0)
print(data[0].shape)
vis.images(data[0],env="main")

```

```python
data_loader = torch.utils.data.DataLoader(dataset = MNIST,
                                          batch_size = 32,
                                          shuffle = False)


for num, value in enumerate(data_loader):
    value = value[0]
    print(value.shape)
    vis.images(value)
    break




vis.close(env="main")

```





### Line plot

```python
Y_data = torch.randn(5)
plt = vis.line (Y=Y_data)

```

```python
X_data = torch.Tensor([1,2,3,4,5])
plt = vis.line(Y=Y_data, X=X_data)

```



#### 라인 업데이트

```python
Y_append = torch.randn(1)
X_append = torch.Tensor([6])

vis.line(Y=Y_append, X=X_append, win=plt, update='append')

```

```python
num = torch.Tensor(list(range(0,10)))
num = num.view(-1,1)
num = torch.cat((num,num),dim=1)

plt = vis.line(Y=torch.randn(10,2), X = num)
#여러라인 on single windows
```

#### Line Info

```python
plt = vis.line(Y=Y_data, X=X_data, opts = dict(title='Test', showlegend=True))

```

```python
plt = vis.line(Y=Y_data, X=X_data, opts = dict(title='Test', legend = ['1번'],showlegend=True))

```

```python
plt = vis.line(Y=torch.randn(10,2), X = num, opts=dict(title='Test', legend=['1번','2번'],showlegend=True))

```



#### 함수로 짜기

```python
def loss_tracker(loss_plot, loss_value, num):
    '''num, loss_value, are Tensor'''
    vis.line(X=num,
             Y=loss_value,
             win = loss_plot,
             update='append'
             )

```

```python
plt = vis.line(Y=torch.Tensor(1).zero_())

for i in range(500):
    loss = torch.randn(1) + i
    loss_tracker(plt, loss, torch.Tensor([i]))

```

#### 종료

```python
vis.close(env="main")
```

