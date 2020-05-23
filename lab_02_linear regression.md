## Part 1 - Basic ML

lab_02 

Linear Regression



```python
#Training with Full Code
#Data
x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[1],[2],[3]])

#Model Initialization
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

#Optimizer Select
optimizer = optim.SGD([W, b] , lr=0.01)

nb_epochs = 1000
for epoch in range(nb_epochs+1):
  hypothesis = x_train * W + b
  cost = (torch.mean(hypothesis-y_train)**2)
  optimizer.zero_grad()
  cost.backward()
  optimizer.step()

  if epoch %100 == 0:
    print('Epoch {:4d}/{} W:{:.3f}, b: {:.3f} Cost: {:.6f}'.format(epoch, nb_epochs, W.item(), b.item(), cost.item()))
```



```python
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])
# 모델 초기화
model = LinearRegressionModel()
# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=0.01)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    
    # H(x) 계산
    prediction = model(x_train)
    
    # cost 계산
    cost = F.mse_loss(prediction, y_train)
    
    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    # 100번마다 로그 출력
    if epoch % 100 == 0:
        params = list(model.parameters())
        W = params[0].item()
        b = params[1].item()
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, W, b, cost.item()
        ))

```

