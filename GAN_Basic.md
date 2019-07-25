# GAN(Generative Adversarial Network)

아래 사진은 2014년에 Ian Goodfellow님이 쓴 ‘Generative Adversarial Nets’라는 논문입니다. <br/>
GAN(Generative Adversarial Network)은 이 논문을 통해 세상 밖으로 나오게 되었습니다. <br/>

<img src="https://github.com/hwk06023/GAN/blob/master/Images/Generative%20Adversarial%20Nets(2014).png" alt="Generative Adversarial Nets(2014)" width="300" height="400"> <br/>

이 문서에서도 GAN의 기초를 다루기 때문에, 이 논문을 중심으로 설명을 하게 될 것 같습니다.<br/>

## Ian Goodfellow

<img src="https://github.com/hwk06023/GAN/blob/master/Images/lan%20Goodfellow.png" alt="lan Goodfellow" width="300" height="300">

https://www.youtube.com/watch?v=HGYYEUSm-0Q <br/>
GAN은 2014년 처음 나왔을 때도 관심을 모았었지만, 이는 시작에 불과했습니다. <br/>
NIPS 2016에서 Ian Goodfellow님이 GAN tutorial 발표를 하고, 반응은 폭발적이였습니다. <br/>
거기에 제 CNN 문서에서 나왔던 Yann LeCun 박사도 이에 10~20 년 사이의 머신러닝 연구 중에서 <br/>
최고의 아이디어라며 찬사를 보냈고, 이에 많은 이들이 GAN을 공부하고 응용하고 발전시켰습니다. <br/>

Ian Goodfellow님은 그 외에도 Deeplearning의 대가 중 한 분인 Yoshua Bengio와 <br/>
Aaron Courville과 함께 아래 사진의 DEEP LEARNING 책을 썼습니다. <br/>
![DEEP LEARNING_book](https://github.com/hwk06023/GAN/blob/master/Images/DEEP%20LEARNING_book.png)

<br/>

Ian Goodfellow님은 GAN의 방식을 설명할 때, 경찰과 위조지폐범으로 비유를 합니다. <br/>
이는 GAN의 특징을 이해하기 쉽고 재밌어서 많은 사람들이 GAN을 설명할 때 쓰는 비유로 자리매김합니다. <br/>

<br/>

먼저 위조지폐범은 실제 지폐와 똑같이 만들어서 경찰이 구별할 수 없게 하는 것이 목적이고, <br/>
경찰은 위조지폐범이 만든 위조 지폐와 실제 지폐를 완벽하게 구별하는 것이 목표입니다. <br/>
이를 정리하면 " __경찰의 구별하는 정확도를 위조 지폐범은 최소화, 경찰은 최대화 하는 것__ "이 목표가 됩니다.

<br/>

<div style="float:left;">
<img src="https://github.com/hwk06023/GAN/blob/master/Images/police.png" alt="police" width="180" height="180"><img src="https://github.com/hwk06023/GAN/blob/master/Images/burglar.png" alt="burglar" width="180" height="180">
<div/>

<br/>

우리는 Dataset과 매우 유사한 패턴을 만드는 것이 목적이니, 위조지폐범과 목적이 같다고 할 수 있습니다! <br/>
여기서 경찰은 진짜와 가짜의 구별을 완벽하게 한다는 가정이 있어야 합니다. 위조지폐범은 반복해서 구별당하며, <br/>
점점 더 진짜 지폐 같은 위조 지폐를 만들어, 결국에는 경찰이 진짜 지폐와 위조 지폐를 구별할 수 없게 합니다. <br/>
구별할 수 없게 된다는 말은 곧 경찰에게 진짜와 가짜의 구별을 시켰을 때, 둘 중 하나를 아무거나 고르게 되는겁니다. <br/>
이를 정리하면 __위조지폐범은 점점 경찰이 구별하는 정확도를 줄여서, 결국은 경찰이 구별할 확률을 50%로 만듭니다.__

<br/>

이를 통해 알 수 있는 내용인 경찰이 진짜와 가짜의 구별을 완벽히 한다는 가정(max)과 <br/>
위조지폐범이 가짜와 진짜의 차이를 최소화(min)하는 과정을 수학적으로 Minmax Game이라고 합니다. <br/>

#### 이제 정말 모델의 구조를 직접 보면서 이해를 해봅시다! <br/><br/>


## Model

![GAN_Architecture](https://github.com/hwk06023/GAN/blob/master/Images/GAN_Architecture.png) <br/>

위 사진은 이해를 돕기 위해 제가 만든 GAN의 구조 이미지입니다. <br/>
Random noise를 Input 으로 받아, Fake image를 Real image(Training set)와 비슷하게 생성해 주는 <br/>
생성자(Generator)와 이러한 Fake image와 Real image를 구별하는 구별자(Discriminator)로 나뉩니다. <br/>
생성자와 구별자가 대립(Adversarial)하며 서로 성능을 점차 개선해 나가는 구조 입니다. <br/><br/>

왜 대립하는 구조라고 부르는 것일까요? 왜냐하면 Discriminator는 Generator가 만든 <br/>
Fake image와 Real image를 더 잘 구별해야 하고, Generator는 점차 Fake image를 <br/>
Real image에 가깝게 만들어 Discriminator가 구별하기 힘들게 해야하기 때문입니다. <br/>
위의 비유에 대입해보면, 위조지폐범은 Generator, 경찰은 Discriminator입니다. <br/>

<br/>

![objective_funtion](https://github.com/hwk06023/GAN/blob/master/Images/objective_funtion.png) <br/>

이 식은 Generative Adversarial Nets (2014) 논문에 나오는 수식입니다. <br/>
이 식은 위에 설명한 내용을 표현한 수식으로, GAN의 목적 함수(objective funtion)입니다. <br/><br/>
 
먼저 제가 만든 이미지를 수식의 이해를 돕기 위해 직관적으로 변경해보았습니다.<br/>

![Architecture_funtion](https://github.com/hwk06023/GAN/blob/master/Images/Architecture_funtion.png) <br/>

### Discriminator : <br/>

<img src="https://github.com/hwk06023/GAN/blob/master/Images/Discriminator_funtion.png" alt="Discriminator_funtion" width="600" height="270"> <br/>

### Generator : <br/>

![Generator_funtion](https://github.com/hwk06023/GAN/blob/master/Images/Generator_funtion.png) <br/><br/><br/>

![x~pdata(x)](https://github.com/hwk06023/GAN/blob/master/Images/x%7Epdata(x).png) <br/>
확률 밀도 함수입니다. 즉 Real image (Training set)의 분포를 x좌표로 Sampling하겠다는 뜻입니다. <br/><br/>

![D(x)](https://github.com/hwk06023/GAN/blob/master/Images/D(x).png) <br/>
0~1을 내보내야 하기 때문에, D(x)가 1일 때, 0으로 최대가 되며, D(x)가 0일 때 음의 무한대가 되어 최소가 됩니다. <br/><br/>

![z~pz(z)](https://github.com/hwk06023/GAN/blob/master/Images/z%7Epz(z).png) <br/>
처음 Input 해주는 Random noise 값입니다. 연속균등분포한 다차원 배열을 Sampling 해줍니다. <br/><br/>

![G(z)](https://github.com/hwk06023/GAN/blob/master/Images/G(z).png) <br/>
Random noise를 Input 받아 Generator가 생성해낸 Fake image입니다. <br/><br/>

![D(G(z))](https://github.com/hwk06023/GAN/blob/master/Images/D(G(z)).png) <br/>
Generator는 최대한 Discriminator가 Fake image를 받았을 때, 최대한 1에 가까운 값을 내놓도록 학습합니다. <br/><br/><br/>

![Graph](https://github.com/hwk06023/GAN/blob/master/Images/Graph.png) <br/>
검은 점선 : Data generating distribution, 파란 점선 : Discriminator distribution <br/>
녹색 선: Generative distribution, 위로 뻗은 화살표 : x = G(z)의 mapping <br/>

(a), (b), (c), (d)는 학습을 반복하면서 점점 진행되는 순서 입니다. <br/>
녹색 선(생성자의 확률분포)과 검은 점선(데이터의 확률분포)이 가까워지며, <br/>
파란 점선(구별자)이 0.5인 상태(D(x) = 0.5[구별 불가])가 됩니다. <br/>

#### 정상적으로 학습이 이루어졌을 때의 모습입니다.

## GAN의 문제점

그럼 이제 GAN이 태생적으로 갖는 문제점들을 알려드리도록 하겠습니다. <br/>

처음에 GAN을 경찰과 위조지폐범 관계로 비유했을 때 Minmax Game이라고 했습니다. <br/>
경찰이 진짜와 가짜의 구별을 완벽히 한다는 가정(max)에서 <br/>
위조지폐범이 가짜와 진짜의 차이를 최소화(min)하기 때문이였는데요. <br/>

실제로 GAN에서 Discriminator가 처음부터 완벽히 구별을 할 수 없기 때문에, <br/>
Minmax Game이 될 수 없는 모순이 존재합니다. <br/>

따라서 경찰이 구별을 제대로 못하면, 위조지폐범이 진짜 지폐와 정말 유사하게 만들지 않아도, <br/>
경찰을 쉽게 속일 수 있게 되어, 위조 지폐는 결코 더 진짜 지폐와 유사해지지 못할 것입니다. <br/>

즉 Dicsriminator은 처음부터 완벽할 수 없기 때문에, Generator과 같이 학습을 하더라도, <br/> 
Generator가 Discriminator가 완벽치 않은 상태에 속이고 학습을 끝나면 성능을 기대하긴 힘들 것입니다. <br/>

### 모델의 진동(Oscillation)

<img src="https://github.com/hwk06023/GAN/blob/master/Images/Oscillation.png" alt="Oscillation" width="300" height="300"> <br/>
Generator와 Discriminator 두 네트워크 모두 단순히 계속 loss만 줄이려고 하기 때문에, <br/>
오래 학습해도 더 이상 원하는 지점으로 수렴하지 않는 현상을 모델이 진동(Oscillation)한다고 합니다. <br/>

### Mode Collapsing

<img src="https://github.com/hwk06023/GAN/blob/master/Images/Mode%20Collapsing.png" alt="Mode Collapsing" width="300" height="300"> <br/>
학습 데이터의 분포가 다양할 때, 정상적으로 학습이 진행된다면 다양한 분포로 학습이 잘 되야하는데, <br/>
Randomnoise가 다양하게 분포된 데이터들에게 골고루 분포되지 않았을 때 문제가 발생합니다.
Mode Collapsing에서의 Mode는 통계학에서의 의미로 최빈값을 의미합니다. <br/>
즉 Mode Collapsing은 다양하게 분포된 데이터 중 일부분에만 분포되어 학습하게 되는겁니다. <br/><br/>

Mode Collapsing의 설명을 돕기 위해 MNIST로 예를 들어 보겠습니다. <br/>
##### (MNIST는 0~9까지의 손 글씨 이미지 데이터입니다.) <br/>

<img src="https://github.com/hwk06023/GAN/blob/master/Images/Trainingset_mnist.png" alt="Trainingset" width="300" height="240"> <br/>
먼저 Training Set(MNIST)의 확률 분포가 다음과 같다고 가정해보겠습니다.<br/>

<img src="https://github.com/hwk06023/GAN/blob/master/Images/Modecollapse_mnist.png" width="650" height="240"> <br/>
처음 Generator가 학습할 Randomnoise가 다음과 같이 하나의 데이터에만 집중될 경우, <br/>
이 데이터로 Generator는 Discriminator가 구별할 수 없는 상태가 되기까지 해당 데이터를 학습합니다. <br/>

<img src="https://github.com/hwk06023/GAN/blob/master/Images/ModecollapsinginMNIST.png" width="150" height="150"> </br>
학습이 순조롭게 진행이 되더라도 이러한 Mode collapsing이 발생하면 우리가 기대했던 다양한 숫자가 아닌 <br/>
처음 Randomnoise가 집중된 한 곳의 데이터만을 Generator가 학습해서 만들어냅니다. <br/>

### 해결 방안

모델이 분포된 전체 데이터들을 골고루 학습하게 도와줍니다. <br/>
Generator와 Discriminator가 같이 서로 잘 학습해 나갈 수 있도록 도와줍니다. <br/><br/>


## 마무리
현재에는 GAN에 대한 많은 아이디어와 개선으로 다양한 모델들이 존재합니다. <br/>
이번에는 그러한 모델들의 기초를 제대로 공부하는 시간이 되었어서 뿌듯합니다. &#128522; <br/>
