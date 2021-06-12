## 🤗 Result

### environment
```
mlflow == 1.15.0
torch == 1.7.1
torchtext == 0.8.1
pytorch-lightning == 1.2.8
```


### result
#### pretraining
- data : [petition data](https://github.com/lovit/petitions_archive)
- task : Language modeling

```
python run_main.py
```
![image](https://user-images.githubusercontent.com/46675408/120097868-c5db1f00-c16d-11eb-91fa-41763c01a640.png)

|train loss|train accuracy|
|---|---|
|0.343|0.55|


#### finetuning
- data : [KLUE](https://klue-benchmark.com/tasks/66/leaderboard/task)
- task : topic classification

```
python run.py
```

##### accuracy
![image](https://user-images.githubusercontent.com/46675408/120183336-82a2ae00-c24a-11eb-8937-3ce061567e93.png)
|train accuracy|valid accuracy|
|---|---|
|0.95|0.79|

##### fscore
![image](https://user-images.githubusercontent.com/46675408/120183572-d614fc00-c24a-11eb-9aa5-5a5069c7bf29.png)
|train f-score|valid f-score|
|---|---|
|0.9319|0.7639|

#### ablation study
. gamma vector added :

![image](https://user-images.githubusercontent.com/46675408/120253013-35155800-c2c1-11eb-943f-23711215fa93.png)
|gamma vector x valid accuracy|gamma vector o valid accuracy|
|---|---|
|0.74|0.79|

gamma vector
![image](https://user-images.githubusercontent.com/46675408/120253141-92110e00-c2c1-11eb-91c7-60dbac14dc57.png)


. chr vs token while finetuning :

![image](https://user-images.githubusercontent.com/46675408/120248286-7867ca80-c2b1-11eb-9688-7bdecb50654a.png)

캐릭터 단위가 오히려 높음 unk 토큰이 많아서 그런가?  

## 🤔 Paper review
![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/068b3b54-4f13-42be-8725-88081a29b6fe/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/068b3b54-4f13-42be-8725-88081a29b6fe/Untitled.png)

**2) (슬랙으로 이미 토론을 했지만 그래도) 이해가 안 가는 부분, 이해가 안 가는 이유(논문 본문 복붙)

=** CNN을 어떻게 했다는 건지? 저 Srivastava 논문이랑 똑같이 하면 되는건가? 2048 character n-gram이라는게 무슨 뜻인지
-> cnn filter가 2048개다 

CNN-BIG-LSTM in Jozefowicz et al. ´(2016). The final model uses L = 2 biLSTM layers with 4096 units and 512 dimension projections and a residual connection from the first to second layer. The context insensitive type representation uses 2048 character n-gram convolutional filters followed by two highway layers (Srivastava et al., 2015) and a linear projection down to a 512 representation.

= 아래 문장이 forward, backward의 토큰 표현과 softmax layer를 공유한다고 하는 것 같은데, softmax layer를 공유한다는거는 마지막 FCN까지를 의미한다는건가?  We tie the parameters for both the token representation (Θx) and Softmax layer (Θs) in the forward and backward direction while maintaining separate parameters for the LSTMs in each direction. 
-> FCN 레이어 공유.

= ELMo fine tuning시 어디까지 ? LM을 한 모든 파라미터를 freeze하고 ELMotask와 xk에 대한 임베딩을 concat한다음에 RNN만 학습되는 건가? 

To add ELMo to the supervised model, we first freeze the weights of the biLM and then concatenate the ELMo vector ELMotask k with xk and pass the ELMo enhanced representation [xk; ELMotask k] into the task RNN

= 3.3, 5.2에서 ELMo를 input뿐 아니라 output에 넣으면 성능이 더 좋아진다는 게 무슨 말일까? 

→ rnn의 output으로 나갈때 hidden에 elmo vector를 concat하겠다

including ELMo at both the input and output layers for SNLI and SQuAD improves over just the input layer, but for SRL (and coreference resolution, not shown) performance is highest when it is included at just the input layer

**3) 재밌었던 부분**

. 표 4에서 glove는 토큰 하나당 표현 하나여서 여러 품사의 토큰을 담지만, ELMo의 표현은 그런 품사 등의 정보까지 표현할 수 있는 것
. 여러 가지 당시 논문들을 짬뽕한 점
. 레이어마다 히든 벡터를 사용해서 단어표현을 한 점. 제목이 처음엔 쌩뚱맞았는데 읽다보니 그렇구나..싶음 

**4) 논문 구현 시 주의해야할 것 같은 부분(논문 본문 복붙)**

. residual connection - LSTM
. L2 regularization
. CNN모델에서 highway 
. ELMo task vector 만드는 부분
. fine-tuning task구하기, 데이터셋 구하기

## 🤫 논문과 다르게 구현한 부분


## 🤭 논문 구현하면서 배운 점 / 느낀 점
