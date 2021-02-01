## 논문 구현하면서 배운 점 / 느낀 점
 
- torch에서 왜 inplace 연산이 필요한지 a = a + 1하면 덮어쓰기 돼서 역전파가 진행이 안됨
- `nn.CrossEntropyloss`가 `nn.LogSoftmax()`와 `nn.NLLLoss()`를 결합한 것이다
- 사용자 데이터셋으로 torchtext dataset 형태처럼 만드는 법
- `torchtext`의 `Example`이 행렬이나 데이터프레임의 row로 사용된다는 점 -> `Example.fromlist(데이터, 필드)`와 같이 사용 가능하다는 것
- torch debugging 방법..모델에서 바꿀게 아니라 간단한 텐서 만들어서 해야 정확하다는 점
- dropout의 train, inference때 작동 차이 
- batch_size의 크기에 상관없이 모델을 만들어야 함
- Adam의 대단함과 Adam이 나오기 전에 있었던 여러가지 정규화 방식들
- max norm regularization, L2 norm regularization의 차이
- `Dataset`에서 하는 처리와 `DataLoader`에서 해야 하는 처리 
 

## Paper review

### 1) PPT 한 장 분량으로 논문 정리
![Untitled](https://user-images.githubusercontent.com/46675408/105627460-706aca80-5e7a-11eb-96e8-187b3faac601.png)


### 2) 이해가 안 가는 부분, 이해가 안 가는 이유

word2vec에 없는 단어를 pre-trained 된 벡터의 variance 와 같도록 U[-a, a]로 뽑는 것

### 3) 재밌었던 부분

good 벡터가 bad 벡터와 word2vec에서 가깝다가 분류 모델 이후엔 멀어진 것.

위와 같은 반의어 관계에서 좋은 워드 임베딩을 구하기 위해서는 CBOW, skip-gram으로는 분명히 한계가 있겠고, 분류모델 등으로 해결 가능하겠다고 생각하게 됨.

### 4) 한계로 느껴지는 부분

max-over-time-pooling이 중요한 feature만 뽑고 다른 길이의 input을 자연스럽게 해결했다고 하는데 max값을 취하면서 정보가 많이 사라졌을 것 같다는 생각이 듦 

### 5) 논문 구현 시 주의해야할 것 같은 부분

- word2vec에 없는 단어는 랜덤 vector로 주어져야함 (Words not present in the set of pre-trained words are initalized randomly) : 보통 패키지에서 없는 단어는 [UNK] 토큰으로 한번에 처리하기 때문에이 부분을 새롭게 구현해야 할듯
- CNN - static : word vector는 고정 → with no grad을 모델 중간에 넣을 수 있나? 해본 적이 없음..
- CNN - mutli-channel : 한 채널은 픽스하고 한 채널은 back prop 되어야 함. 이것도 no grad ?
- L2 norm weight : ??? in place operation
- gradient clip
- CNN max-over-time-pooling

### 6) 같이 얘기해봤으면 하는 부분

CNN 부분이 RNN 계열이었다면 성능이 더 높게 나올까? → vanillaRNN 다음 실험해보자  

- (낮게 나올거라고 생각한다면) 왜 그럴까? '언어 문제를 time-series로 본다'는 것을 구현하려면 어떤 방법이 더 적합할까? 혹은 그냥 장단이 있는걸까?

### 7) (페이퍼 리뷰 후에) 느낀점. 새로 알게 된 점

- CNN이 강조된 논문이라고 생각했었는데 pretrained된 word2vec 모델이 universal 하게 작동하다는 것에 좀 더 강조를 둔 논문이란 것을 알게 됨
- 이 논문이 언어에서 처음 CNN이 작동한다는 사실을 알게 된 논문은 아님(CNNs models have subsequently been shown to be effective for NLP ...)
- max-over-time-pooling 이라는 용어. time series 차원에서 max 하는 것을 뜻함.
