## 🤗 Result
WIP

## 🤔 Paper review

**1) PPT 한 장 분량으로 자유롭게 논문 정리 뒤 이미지로 첨부**
![image](https://user-images.githubusercontent.com/46675408/121776733-7c58ee00-cbc9-11eb-8c31-ede9ffc29e88.png)


**2) (슬랙으로 이미 토론을 했지만 그래도) 이해가 안 가는 부분, 이해가 안 가는 이유(논문 본문 복붙)**

ㄱ. 3.2.3 self attention을 사용하면 previous layer의 output이 K, V, Q가 된다는건가

The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the **previous layer** in the encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder 

self-attention + FFN이 몇층으로 쌓는건데 한 layer의 output이 어떻게 되는거지? $$d_k$$차원 짜리로 벡터가 그 자체로 또 K, V, Q가 되는건가?

→ FFN 후의 (seq_len, d_model)의 MATRIX을 각 K, V, Q로 LINEAR PROJECTION해서 재사용

ㄴ. Encoder에서도 stack layer가 N = 6 이고 decoder 에서도 stack layer가 N = 6인데 우리의 역사적인 토론 주제인 인코더 디코더는 각 stack에서 히든벡터로 연결되는가 아님 인코더의 마지막 stack만 가는가 가 궁금하네요 wikidocs는 후자처럼 그려지네용

→ 맨 위 stack만 가는게 맞다
ㄷ.  3.2.3 마이너스 무한대로 뭘 어떻게 바꿨다는건지..? attention 값을 바꾼건가..? [MASK] 이런 토큰으로 바꾸면 왜 안될까?

Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position. We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of **scaled dot-product attention by masking out (setting to −∞)** all values in the input of the softmax which correspond to illegal connections. See Figure 2.

→ 정방행렬로 나오는 attention value를 위 직각삼각형을 마이너스 무한대로 바꾼듯 그래야 softmax 값이 0이 됨

→[MASK] token으로 바꾸면 어찌됐든 [MASK]란 토큰이 들어간 채로 학습이 될거여서 아예 어텐션 벨류를 마이너스 무한대로 해서 softmax를 0으로 바꿔서 학습이 안되게 하기

ㄹ. 3.4. share embedding?

In our model, we share the **same weight matrix between the two embedding layers** and the pre-softmax linear transformation, similar to [30]. In the embedding layers, we multiply those weights by √dmodel. ...?

ㅁ. label smoothing 
크로스 엔트로피에서 정답 레이블을 1, 0 으로 두는게 아니고  1 - 엡실론, 0 + 엡실론으로 두는 것
1) 오버피팅을 방지할 수 있음
모델이 트레이닝 데이터의 GT 레이블에 full probability를 할당하면, 일반화하기 어려울 수 있음
2) 가장 큰 logit과 작은 logit의 차이를 크게 만들면, 모델이 adapt할 능력을 줄임

**3) 재밌었던 부분**
. additive attention보다 dot-product가 행렬연산이기 때문에 더 효율적이라는 점....생각지도 못함...
. self attention 이 여기서 처음 나온 건 아니구나 이 논문이 나오기 전에 나왔던 유사한 시도를 한 무수히 많은 논문이 있구나..
. stack은 높이로 올라가는거고, multi-head는 두께라고 생각하면 이 모델은 참 높이도 쌓았고 두껍게도 쌓았구나 여러 차원으로 많이 쌓았네 어떻게 보면 CNN 쌓는 느낌이랑도 비슷하다
. self attention의 장점을 논리적으로 쓴 부분

**4) 논문 구현 시 주의해야할 것 같은 부분(논문 본문 복붙)**

. scaled dot attention<br>
. self attention<br>
. stack self attention?<br>
. Residual block<br>
. multi-head self attention<br>
. masking<br>
. positional encoding<br>
. optimizer<br>
. warm-up step 

## 🤫 논문과 다르게 구현한 부분
- dataset : multi30k

## 🤭 논문 구현하면서 배운 점 / 느낀 점
- transformer 구조에 대한 이해 : encoder에서의 query가 decoder로 넘어가는 부분을 정확히 이해 못했는데 마지막 layer의 attention value를 넘겨서 이를 K, V를 곱해서 구하는거구나 이해를 함.
- `matmul`과 `bmm`의 차이 : matmul은 broadcasting 처리 됨
- stacked self-attention을 구현하기 위해 FCN에서 linear로 한번 차원을 키웠다가 다시 낮춤
- multi-head self-attention을 하기 위해 Q, K, V별로 head_dim을 만드는게 아니라 hid_dim을 만들고 n_heads 만큼 자르는 방식
- label smoothing cross entropy 트릭과 구현
- `nll_loss`의 `ignore_index`의 의미
- `register_buffer`의 사용 이유 : `model.parameters`에 안나오게 하려고
- `LambdaLR`의 사용 : scheduler는 optimizer의 lr에 `*=` 연산이 되기 때문에 optimizer에서 lr은 1로 해줘야함 
- broadcasting을 사용한 src, trg에 대한 mask 구현 방식
- `scheduler.step()`이 `optimizer.step()` 뒤로 와야 함
