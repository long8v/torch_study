## 🤗 Result
🚩 데이터셋이 다름(Multi 30k)
|model|maxout|# of parameters|test PPL|test BLEU|training time for one epoch|
|----|----|----|----|----|
|reference code 그대로|x|21,196,869|13.162|39.637|3m 15s~3m 20s|
|referecne code + maxout|o|14,631,921|12.380|40.204|3m 2s~4m 40s|
|논문 파라미터 w/ maxout|o|40,127,409|12.747|40.328|4m 12s~4m 16s|


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

## 🤭 논문 구현하면서 배운 점 / 느낀 점
- aligning이라는 용어
- Baddhanau attention
- [maxout](https://m.blog.naver.com/PostView.nhn?blogId=laonple&logNo=220836305907&proxyReferer=https:%2F%2Fwww.google.com%2F)
- [orthgonal initialization](https://smerity.com/articles/2016/orthogonal_init.html)
