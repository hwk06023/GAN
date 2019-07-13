# GANs(Generative Adversarial Networks)

GANs는 2014년, ‘Generative Adversarial Nets’이라는 논문으로 세상에 나오게 되었습니다. <br/>
(여담으로 논문을 쓰신 Ian Goodfellow님은 아래 DEEP LEARNING 책을 쓰신 분이기도 합니다.) <br/>

![DEEP LEARNING_book](https://github.com/hwk06023/GAN/blob/master/Images/DEEP%20LEARNING_book.png)

<br/>

다시 본론으로 돌아가 GAN(Generative Adversarial Network)은 비지도 학습의 대표적인 모델로, <br/>
최근까지 GAN을 활용한 굉장히 다양한 모델들이 나오고 있습니다. 이러한 GAN에 대해 자세히 알아봅시다 !! <br/><br/>

![GAN_Architecture](https://github.com/hwk06023/GAN/blob/master/Images/GAN_Architecture.png) <br/>

위 사진은 이해를 돕기 위해 제가 만든 GAN의 구조 이미지입니다. <br/>
Random noise를 Input 으로 받아, Fake image를 Real image(Training set)와 비슷하게 생성해 주는 <br/>
생성자(Generator)와 이러한 Fake image와 Real image를 구별하는 구별자(Discriminator)로 나뉩니다. <br/>
생성자와 구별자가 대립(Adversarial)하며 서로 성능을 점차 개선해 나가는 구조 입니다. <br/><br/>

왜 대립하는 구조라고 부르는 것일까요? 왜냐하면 Discriminator는 Generator가 만든 <br/>
Fake image와 Real image를 더 잘 구별해야 하고, Generator는 점차 Fake image를 <br/>
Real image에 가깝게 만들어 Discriminator가 구별하기 힘들게 해야하기 때문입니다. <br/><br/>

![objective_funtion](https://github.com/hwk06023/GAN/blob/master/Images/objective_funtion.png) <br/>

이 식은 Generative Adversarial Nets (2014) 논문에 나오는 수식입니다. <br/>
이 식은 위에 설명한 내용을 표현한 수식으로, GAN의 목적 함수(objective funtion)입니다. <br/><br/>
 
먼저 제가 만든 이미지를 수식의 이해를 돕기 위해 직관적으로 변경해보았습니다.<br/>

![Architecture_funtion](https://github.com/hwk06023/GAN/blob/master/Images/Architecture_funtion.png) <br/>

### Discriminator : <br/>

![Discriminator_funtion](https://github.com/hwk06023/GAN/blob/master/Images/Discriminator_funtion.png) <br/>

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
Generator는 최대한 Discriminator가 Fake image를 받았을 때, 최대한 1에 가까운 값을 내놓도록 학습합니다. <br/><br/>




