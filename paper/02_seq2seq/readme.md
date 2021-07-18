## 🤗 Result
WIP

## 🤔 Paper review
**1) PPT 한 장 분량으로 자유롭게 논문 정리 뒤 이미지로 첨부**
![Untitled (1)](https://user-images.githubusercontent.com/46675408/108203501-f15d6f00-7165-11eb-9c68-8da61962b002.png)

**2) (슬랙으로 이미 토론을 했지만 그래도) 이해가 안 가는 부분, 이해가 안 가는 이유(논문 본문 복붙)**

= 우리의 objective function는 ? 

계속 말하고 있는 perplexity가 모든 t시점의 softmax 결과값인 logSoftmax loss를 다 더한것이겠지? → ㅇㅇ 맞다<br>
= 논문에서 임베딩 시각화한 것 : Thus the deep LSTM uses 8000 real number to represnt a sentence에서 8000은 1000(=hidden cell dim) * 4(=num layers of LSTM)  * 2(=hidden, cell state) 인건 알겠는데 PCA할때 그냥 concat했으려나 아님 (1000 by 4 by 2) 를 PCA? 후자일듯?

= SMT에서 rescore?

SMT 에서 나온 결과물을 개선시키기 위해 SMT에서 뽑힌 1000개의 리스트를 재정렬하는 것

Finally, we used the LSTM to rescore the publicly available 1000-best lists of the SMT baseline on
the same task [29]. By doing so, we obtained a BLEU score of 36.5, which improves the baseline by
3.2 BLEU points and is close to the previous best published result on this task (which is 37.0 [9]).

**3) 재밌었던 부분**

= reversed : 나중에 RNN seq2seq쓰는 것 있음 실험해봐야겠다 싶었음! 

= PCA해봤더니 단어는 거의 비슷한데 의미는 반대인 것이 묶인 것. 결과가 꽤 놀라워서 체리피킹인지 아닌지 꼭 테스트를 해봐야겠다

= softmax 구하는데만 4개의 GPU를 쓴 것...ㅋㅋ→ 그럼 요즘은 다 hierachial softmax 쓰는건가?

= size가 1인 beam search 즉 greedy search 성능이 나쁘지 않았던 것. 내가 seq2seq + greedy search를 썼을 땐 계속 확률이 높은 것 같은 똑같은 단어를 반복하던데 그런건 BLEU에서 크게 penalty 되지 않아서일까? 아니면 내가 트레이닝을 이상하게 시켜서 그런걸까 

= We found deep LSTMs to significantly outperform shallow LSTMs, where each additional layer reduced perplexity by nearly 10% → 더 깊은게 항상 좋은건 아닌데 이 경우엔 깊은게 훨씬 좋았다네..그냥 신기

**4) 논문 구현 시 주의해야할 것 같은 부분(논문 본문 복붙)**

= most frequent 단어만 사용하고 나머지는 [UNK] 처리함 → 결국 corpus 한 바퀴 다 봐야함ㅎㅎ
We used 160,000 of the most frequent words for the source language
and 80,000 of the most frequent words for the target language. Every out-of-vocabulary word was
replaced with a special “UNK” token.

= LSTM weight uniform 초기화
We initialized all of the LSTM’s parameters with the uniform distribution between -0.08
and 0.08

= beam search decoder 
We search for the most likely translation using a simple left-to-right beam search decoder

= exploding gradient를 피하기 위하여 L2 norm gradient clipping
Thus we enforced a hard constraint on the norm of the gradient by scaling it when its norm exceeded a threshold.

= 비슷한 길이 애들끼리 묶어줘야함! 
To address this problem, we made sure
that all sentences in a minibatch are roughly of the same length, yielding a 2x speedup.

## 🤫 논문과 다르게 구현한 부분
- dataset 
- optimizer
- halving learning rate every half epoch
- beam search 미구현..

## 🤭 논문 구현하면서 배운 점 / 느낀 점
- [bucketing이 뭔지](https://stackoverflow.com/questions/49367871/concept-of-bucketing-in-seq2seq-model)(bucketiterator가 단순히 길이 순으로 정렬해주는 것뿐 아니라 bucketing이라는 연산까지 해준다는 점)
- `pack_padded_sequence`, `pad_packed_sequence` : 배치로 묶을 때 zero-padding이 생기고 RNN이 해당 zero-padding을 굳이 거치지 않게 하는 것이 [packing](https://simonjisu.github.io/nlp/2018/07/05/packedsequence.html)
- `.to(device)`는 모델에 데이터 부을때 넣는게 가장 효율적이다
- torchtext의 `Field` 구현해 봄
- collections의 namedtuple 매우 유용(DataLoader 구성할 때 `.src` 접근하려고 사용함) 
- Iterator의 sort, sort_key, sort_within_batch argument
- torch의 `nn.LSTM`의 input output shape. for 문으로 hidden, cell 안넣어 줘도 모든 시퀀스에 대해 recurrent 계산을 해줌
- multi-layered LSTM의 encoder-decoder 연결하는 cell, hidden이 모든 layer에서 연결되도록 구현되어 있다는 점
- seq2seq의 decoder를 학습할 때에는 `<eos>`토큰이 들어가면 안됨 
- seq2seq의 encoder는 `<sos>` 토큰을 안 넣어도 됨
- teacher force이 코드 상 어떻게 구현되는지
