## 🧐 run
.en, .fr이 라이브러리로 못 불러와서 [홈페이지](https://github.com/multi30k/dataset/tree/master/data/task1/raw) 들어가서<br>
train.en, train.fr, val.en, val.fr 다운 받아서 실행 경로에 생기는 .data(숨김폴더)에 들어가서 넣어줘야 함

- script
```
python main.py
```
- notebook
ㄴ 01_reference_code.ipynb : 데이터셋 바꿔서 영어 튜토리올 한글로 번역해 봄
ㄴ 02_reference_code_paper_detail.ipynb : 논문과 비교해보고 추가로 디테일 구현해 봄

## 🤗 Result
🚩 데이터셋이 논문과 다름(Multi 30k en-fr)
|model|maxout|# of parameters|test PPL|test BLEU|training time for one epoch|
|----|----|----|----|----|----|
|reference code 그대로|x|21,196,869|13.162|39.637|3m 15s~3m 20s|
|referecne code w/o maxout|o|14,631,921|12.380|40.204|3m 2s~4m 40s|
|논문 파라미터 w/ maxout|o|40,127,409|12.747|40.328|4m 12s~4m 16s|

maxout을 사용하면 파라미터 크기 대비 성능이 좋으나, max연산 때문인지 속도는 오히려 느려졌다

## 🤔 Paper review
**1) PPT 한 장 분량으로 자유롭게 논문 정리 뒤 이미지로 첨부**
![image](https://user-images.githubusercontent.com/46675408/112748663-3348c300-8ff8-11eb-860a-dbdc3e0dbad5.png)

**2) (슬랙으로 이미 토론을 했지만 그래도) 이해가 안 가는 부분, 이해가 안 가는 이유(논문 본문 복붙)**

1) ![image](https://user-images.githubusercontent.com/46675408/112748684-596e6300-8ff8-11eb-8f76-40a94163583a.png)

2) alignment의 FCN부분 + decoder 부분 <br>
  a) encoder의 hidden state에 W벡터를 곱하고, decoder의 hidden state에 U벡터를 곱해서 더한 뒤(concat후 FCN한거랑 같음) tan를 구하고 이를 다시 v로 곱한걸 softmax취한게 attention score..어마어마하군<br>
  b) 이 attention score를 encoder의  hidden state와 곱해서 context벡터를 구한다<br>
  c) context 벡터와 이전 시점의 output vector를 임베딩한거랑 decoder의 히든벡터랑을 weighted sum해서 target이 나오게 된다 

**3) 재밌었던 부분**

성능 그래프 
![image](https://user-images.githubusercontent.com/46675408/112748689-625f3480-8ff8-11eb-85d5-9f67c0bdcf05.png)
1) 전반적으로 성능이 더 좋은 것 →이건 파라미터가 더 많아서 그럴 수 있음
2) 단어 30개까지 학습한 것과 50개까지 학습하는걸 본 다음에 이걸 test셋을 또 sentence lentgh로 평가한 점 →참 훌륭하게 성능평가를 했다..논문 쓰려면 이렇게 해야되는구나
3) RNNsearch-50의 우수함.. 왜 rnn-30은 길이 30 가까이서 떨어지는 추세가 보이는데 50은 저렇게 훌륭할까

alignment function과 RNN function을 본문에선 아주 일반적으로 작성하고 후에 우리는 자유롭게 선택하기 위해 제너럴하게 작성했다고 했다는 점
→ 후에 scaled-dot attention 등 나오게된 초석..?..역시 상상력을 풍부하게 하는게 중요하군

**4) 논문 구현 시 주의해야할 것 같은 부분(논문 본문 복붙)**

. Alignment model
. decoder..
. bi-directional RNN
. attention score를 구하는 것이 scaled-dot product가 아닌 별도의 FCN 
. maxout 
. 단어의 max len 가지고 model 4개 만드는거 
. 각종 initialize

**5) 소개하고 싶은 개념 하나 (발표 5분 분량, 선택)**

[maxout](https://arxiv.org/pdf/1302.4389.pdf)
Dropout의 효과를 극대화시키기 위한 활성화 함수

## 🤫 논문과 다르게 구현한 부분
- dataset : Multi30k english-french
- optimizer : Adam
- initialize 일부
  - $W_a$와 $U_a$는 N(0, 0.001^2)이고 bias는 0 -> 코드에서 concat되어 있는데 그냥...하나로..
  - $V_a$는 다 0으로 초기화 -> $v_a$라고 일단 생각함
- 논문에서 Maxout hidden layer를 사용하는 것과 같은 의미다..라고 쓴걸 Maxout으로 구현함 

## 🤭 논문 구현하면서 배운 점 / 느낀 점
- aligning이라는 용어
- Baddhanau attention
- [maxout](https://m.blog.naver.com/PostView.nhn?blogId=laonple&logNo=220836305907&proxyReferer=https:%2F%2Fwww.google.com%2F) 개념과 이차함수 근사 경험
- [orthgonal initialization](https://smerity.com/articles/2016/orthogonal_init.html) 
- torchtext Field의 `.preprocess`와 `.process`의 존재
- `predict`를 지난 달보다 더 깔끔하게 구현함
- RNN의 ouputs 중 output과 hidden에서 output이 모든 t시점의 마지막 층의 hidden state 를 모아놓은 것이라는 것[.](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html) 
- bi-directional LSTM의 output의 형태(hidden[-1, :, :]이 마지막 단어를 본 forward hidden state이고 hidden[-2, :, :]이 첫번째 단어를 본 backward hidden state
- seq2seq에서 encoder를 bi-LSTM을 썼을 경우 forard, backward의 hidden state를 concat해서 넣어주는 것이 [정석](https://towardsdatascience.com/understanding-bidirectional-rnn-in-pytorch-5bd25a5dd66)
- v 벡터 따로 x 벡터 따로 해서 + 하는 것 대신 v벡터 x를 concat해서 FCN하는 trick
- torch에서 여러 모델을 조립했을 때 `model.named_parameters()`가 얼마나 아름답게 나오는지 
![image](https://user-images.githubusercontent.com/46675408/113498443-e446e480-9547-11eb-9be0-a910635c61c7.png)  
