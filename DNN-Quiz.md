## Part 2 - DNN

### Project A - Quiz





1.다음 중 Python의 Library 및 Package에 대한 설명으로 **옳지 않은 것**은?


numpy : Scientific computing과 관련된 여러 편리한 기능들을 제공해주는 라이브러리이다.

**torch : 최근 2.0버전으로 업데이트 되었으며, 내장되어 있는 Keras 를 이용하여 High-level 구현이 가능하다.**

torch.utils.data : Mini-batch 학습을 위한 패키지이다.

torchvision : PyTorch 에서 이미지 데이터 로드와 관련된 여러가지 편리한 함수들을 제공하는 라이브러리이다.

matplotlib.pyplot : 데이터 시각화를 위한 다양한 기능을 제공하는 패키지이다.



2.**torch.Tensor** 에 대한 설명으로 다음 중 **옳지 않은 것**은?

다차원 배열(Multi-dimensional Matrix)을 처리하기 위한 가장 기본이 되는 자료형(Data Type)이다.

NumPy의 ndarray와 거의 비슷하지만, GPU 연산을 함으로써 computing power를 극대화 시킬 수 있다.

기본텐서 타입(Default Tensor Type)으로 32비트의 부동소수점(Float) torch.FloatTensor으로 되어있다.

**torch.Tensor의 사칙연산은 torch.Tensor 간이나 torch.Tensor와 Python의 Scala 값 뿐만 아니라 Numpy의 ndarray와도 연산이 가능하다.**

NumPy의 ndarray 와 마찬가지로 브로드캐스팅(Broadcasting) 적용이 가능하다.



3.아래 코드는 간단한 Linear Regression을 구현한 것이다.

[![img](https://cphinf.pstatic.net/mooc/20190617_125/1560738140051w6T49_PNG/torch_quiz3.png)](https://cphinf.pstatic.net/mooc/20190617_125/1560738140051w6T49_PNG/torch_quiz3.png)

 

다음 중 **옳지** **않은** 것은?

**#1**(1번 주석, 10행)에서 requires_grad=True 는 학습할 것이라고 명시함으로써, gradient를 자동적으로 계산하라는 뜻이다.

**#2**(2번 주석, 23행)는 cost를 계산하는 과정으로, cost = F.mse_loss(hypothesis, y_train) 또는 cost = nn.MSELoss(hypothesis, y_train)으로 대체될 수 있다.

**#3**(3번 주석, 25행)는 PyTorch에서 backpropragation 계산을 할 때마다 gradient 값을 누적시키기 때문에, gradient 를 0으로 초기화 해주기 위한 것이다.

**#4**(4번 주석, 26행)는 gradeint를 계산하겠다는 의미이다.

**#5(5번 주석, 27행)는 다음 epoch으로 넘어가라는 뜻이다.**





4.로지스틱 회귀 모델(Logistic Regression Model)을 이용하여 이진분류모델(A Binary Classifier)를 만들려고 한다. 

[![img](https://cphinf.pstatic.net/mooc/20190611_6/1560226646623XdNaT_PNG/torch_quiz4.png)](https://cphinf.pstatic.net/mooc/20190611_6/1560226646623XdNaT_PNG/torch_quiz4.png)

다음 중 모델의 손실 함수(Loss Function)와 출력 층(Output Layer)의 활성화 함수(Activation Function)의 최적의 조합으로 알맞은 것은? **4 - Sigmoid and BCELoss**

[![img](https://cphinf.pstatic.net/mooc/20190821_272/1566373921804UvuKl_PNG/torch_quiz4-0.png)](https://cphinf.pstatic.net/mooc/20190821_272/1566373921804UvuKl_PNG/torch_quiz4-0.png)

------

5.다음과 같이 512개의 hidden layer 뉴런으로 구성된 Multilayer Perceptron(MLP) 모델을 이용하여, 28*28의 MNIST 데이터를 10개의 class로 구분하는 모델을 만드려고 한다.

[![img](https://cphinf.pstatic.net/mooc/20190611_167/1560228846254zKH40_JPEG/torch_quiz5-MLP.jpg)](https://cphinf.pstatic.net/mooc/20190611_167/1560228846254zKH40_JPEG/torch_quiz5-MLP.jpg)

[![img](https://cphinf.pstatic.net/mooc/20190617_238/15607677566822N4tE_PNG/_2019-06-17__7.35.34.png)](https://cphinf.pstatic.net/mooc/20190617_238/15607677566822N4tE_PNG/_2019-06-17__7.35.34.png)

위 코드에서 정수 N, M, H에 들어갈 숫자의 조합으로 알맞은 것은? Why ?

- N: 28*28, M: 10, H: 512
- **N: 10, M: 28*28, H: 512**
- N: 10, M: 512, H: 28*28
- N: 28*28, M: 512, H: 10



6.신경망 모델링을 위한 Hyperparameter 에 대한 설명으로 **옳은 것**은?

- Epoch : 하나의 Mini-batch를 한 번 훈련시켰을 때, 1 Epoch을 돌았다고 한다. 

- ->전체 데이터를 모두 학습 1회 시.

- **Batch Size : 하나의 Mini-batch의 크기(Mini-batch의 데이터 개수). 즉, 전체 Dataset 크기를 Mini-batch의 개수로 나눈 것.**

- Iteration : 전체 Training Dataset을 한 번 훈련시켰을 때, 1 Iteration을 돌았다고 한다.

- ->1Epoch 돌았다고 한다.

- Learning Rate : Learning Rate가 클수록 Local Minimum에 빠질 위험이 크다

  -> GLobal Minimum에 빠질 위험이 크다..

7.다음에 대해 맞으면 True 틀리면 False를 선택하시오.

"우리가 모델을 학습시킬 때, 데이터가 많을수록 더 좋은 모델이 된다. 하지만, 많은 양의 데이터를 가져와 한 번에 학습시키기엔 너무 느리고 저장하는 데에 문제가 있다. 따라서, 이를 효율적으로 학습시키기 위해 전체 데이터를 균일하게 Mini-Batch로 나누어서 학습하게 되고, 이 때 overfitting을 막아주기 위해 Epoch 마다 데이터가 학습되는 순서를 바꾸어 준다."

**True**



8.예측 모델에 대한 결과로, (B)와 같은 결과를 가져왔다. 이를 해결하여 (A)와 같이 더 일반화된 예측을 위해 취할 수 있는 방법으로 알맞은 것을 모두 고르시오.

[![img](https://cphinf.pstatic.net/mooc/20190611_159/15602302918508qypC_PNG/torch_quiz8.png)](https://cphinf.pstatic.net/mooc/20190611_159/15602302918508qypC_PNG/torch_quiz8.png)

- Data 양을 더 줄인다.
- Feature의 숫자를 늘린다.
- **Early Stopping: Validation Loss가 더 이상 낮아지지 않을 때 학습을 중단한다.**
- **활성화 함수(Activation Function) 앞에 Batch Normalization을 적용시켜준다.**
- **Dropout 방법을 적용시켜준다.**

->오버피팅시 해결방안 (얼리스타핑, 배치노멀라이제이션, 드랍아웃)



9.다음에 대해 맞으면 True 틀리면 False를 선택하시오.

"비선형의(Non-linear) 활성화 함수(Activation Function)를 쓰는 이유는, 활성화 함수가 선형(Linear)일 경우 퍼셉트론(Perceptron)이 여러 개라도, 즉, 층(Layer)을 깊게 쌓더라도 층이 한 개인 경우와 차이가 없기 때문이다."

**True**

10.다음에 대해 맞으면 True 틀리면 False를 선택하시오.

"학습된 모델의 성능을 Test할 때, torch.Tensor의 requires_grad를 True로 만들어 gradient를 계산하여 업데이트를 해주어야 한다."

**False**, Train시 requires_grad = True설정.