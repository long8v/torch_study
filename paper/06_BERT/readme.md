## 🤗 Result
### Environment
```
mlflow == 1.15.0
torch == 1.7.1
torchtext == 0.8.1
pytorch-lightning == 1.2.8
tokenizers == 0.9.3
pytorch-crf == 0.7.2
Korpora == 0.2
```

### pretraining
**1) run**

1-1) data loading

Korpora에서 데이터를 다운 받아 문서 단위의 list로 피클링하는 코드입니다
코드 내에 소스 파일 저장하는 경로를 바꿔주어야 합니다 
```
python source/load_korpora.py
>>> yes (혹은 원하는 문서 개수만큼의 숫자)
```

1-2) bert 학습을 위한 데이터셋 / tokenizer 학습

문서 단위로 train / valid를 나누고 NSP prediction을 할 때 문서 맨 끝 문장과 다음 문서 맨 첫 문장이 NSP=1으로 예측되면 안되므로, 문장 끝에 [EOD]라는 토큰을 추가합니다. <br>
토크나이저의 경우 wordpiece로만 학습하게되면 한국어의 경우 조사 등 다빈도 토큰이 붙는 경우가 많아 mecab으로 형태소로 자르고 잘린 부분에 ##을 붙인 다음 tokenizers의 `wordpieces_prefix='##'`  argument로 학습을 진행하면 이를 막을 수 있게 됩니다.
```
python source/tokenizer.py
```
저장 경로에 tokenizer 모델이 저장되고 이 파일이 vocab 수, 문장->인덱스 딕셔너리, 인덱스->문장 딕셔너리, 스페셜 토큰등을 담고 있기 때문에 이후 간편하게 진행할 수 있습니다.

1-3) config.yaml 수정
```
data:
    src: '/home/long8v/torch_study/paper/file/bert/bert.txt' # 학습데이터
    src_valid: '/home/long8v/torch_study/paper/file/bert/bert_valid.txt' # validataion 데이터
    vocab:  '/home/long8v/torch_study/paper/file/bert/vocab.json' # vocab 객체
    max_len: 128 # 최대 토큰 개수
    nsp_prob: 0.5   # nsp = 1일 확률(0.5이면 nsp=1, 0 비율이 1:1이라는 것) 
    mask_ratio: 0.1 # MLM을 위한 마스킹 토큰 비율
    batch_size: 64 
model:
    hid_dim: 256
    n_layers: 2
    n_heads: 8
    pf_dim: 512
    dropout: 0.5

train:
    n_epochs: 1000
    device: 'cuda'
    lr: 0.0005
    scheduler: True
    warmup_steps: 100
    train_mlm: True   # mlm 학습 여부 
    train_nsp: False  # nsp 학습 여부
```

1-4) 실행
```
python run.py
```
실행을 하게 되면 mlflow + pytorch-lightning 프레임워크에서 학습이 됩니다. 
위 파일 실행 경로에서 아래 코드를 실행하면 학습 결과들을 웹으로 볼 수 있습니다.
```
mlflow ui
```

모델 학습 metric, checkpoint는 `mlruns/0`폴더 내에서 볼 수 있습니다. [예시](https://github.com/long8v/torch_study/tree/master/paper/06_BERT/bert_example/4035dd4c47fe43c6a507c0d74365211b)


**2) data**
Korpora에서 제공하는 한국어데이터를 사용했습니다.
[petetion data](https://github.com/lovit/petitions_archive), [namu-wiki data](https://github.com/lovit/namuwikitext)

**3) model**

```
  | Name          | Type             | Params
---------------------------------------------------
0 | encoder       | Encoder          | 4.1 M 
1 | nsp           | Linear           | 514   
2 | mlm           | Linear           | 3.0 M 
3 | criterion_nsp | CrossEntropyLoss | 0     
4 | criterion_mlm | CrossEntropyLoss | 0     
---------------------------------------------------
7.0 M     Trainable params
0         Non-trainable params
7.0 M     Total params
28.101    Total estimated model params size (MB)
```
transformer 구조 자체는 4월 레퍼런스 코드였던 [transformer](https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb)

**4) result**

|metric|train|valid|
|:---:|:---:|:---:|
|MLM loss|2.892|4.854|
|NSP loss|0.221|1.067|
|MLM accuracy|0.46|0.40|
|NSP accuracy|0.88|0.55|

**5) experiment**

5-1) scheduler
![image](https://user-images.githubusercontent.com/46675408/127120275-6b6dc8d4-1d85-4a80-afe6-28bf65db5906.png)

### finetuning
**1) task**

BERT의 성능을 측정하기 위한 finetune-task로 NER를 선택하였습니다

**2) run **
2-1) 
```
python run_finetune.py
```

**3) data**

KLUE 벤치마크의 NER 데이터를 사용했습니다

[KLUE NER](https://github.com/KLUE-benchmark/KLUE/tree/main/klue_benchmark/klue-ner-v1)

**4) model size**

```
  | Name    | Type    | Params
------------------------------------
0 | encoder | Encoder | 4.1 M 
1 | fcn     | Linear  | 3.6 K 
2 | crf     | CRF     | 224   
------------------------------------
4.1 M     Trainable params
0         Non-trainable params
4.1 M     Total params
16.222    Total estimated model params size (MB)
```

**5) result**

|metric|train|valid|
|:---:|:---:|:---:|
|loss|13.49|61.15|
|micro F1|0.991|0.922|
|macro F1|0.931|0.791|

**6) experiment**

![image](https://user-images.githubusercontent.com/46675408/127128705-32bca8f5-f099-492d-85be-d5bd7b51d3e6.png)


## 🤔 Paper review
**1) PPT 한 장 분량으로 자유롭게 논문 정리 뒤 이미지로 첨부**



**2) (슬랙으로 이미 토론을 했지만 그래도) 이해가 안 가는 부분, 이해가 안 가는 이유(논문 본문 복붙)**<BR>
. 지난 달 토론 주제 : BERT는 문맥에 따라 같은 **토큰의 표현 값**이 달라지는가?<br>
token embedding → 같겠죠<br>
positonal embedding → 달라지겠죠<br>
segment embedding → 달라지겠죠<br>
======================<br>
attention 하고 난 뒤에는 → 달라지겠죠

달라진다.
  
. MLM을 정확히 어떻게 하는건지 모르겠음 mask 토큰 넣은채로 인풋에 넣은 뒤에 mask 토큰에 대해서만 predict 하게 하려면 어떻게 구현해야하지?<br>
. [MASK] 토큰을 15% 선정하고 80%은 바꾸고 10%은 치환하고 **10%는 그냥 둔거..** 10% 그냥 두는게 의미가 뭐지 걍 둔건가..<BR>

original :    i go to shcool.<BR>
inference :    i [mask] to **[mask].** 15%<BR>
input :        **i [mask] to school.**<BR>
. 선정된 [MASK]들의 prediction을 할 때 그냥 둔 토큰의 임베딩이 들어감

**3) 재밌었던 부분**<BR>
. MLM 아이디어 간단하고 직관적임. <BR>
. BERTbase가 GPT랑 파라미터 개수 맞춰서 나온 것인 것..ㅎㅎ<BR>
. ELMo 얘기 나오는데 왠지 고향친구 만난듯한 반가움..<BR>
. **feature base**랑 **fine-tuning**이랑 깔끔하게 나눠준거. 항상 fine-tuning 얘기하면 논란이 되는 부분이었음.<BR>
. 그리고 또 feature base로 문제를 풀어주심<BR>
. 데이터를 문장 단위로 잘라서 SHUFFLE한 데이터셋보다는 DOCUMENT 단위의 데이터셋을 써야지 긴 문맥을 뽑아낼수 있어서 좋다고 함.<BR>
  
**4) 논문 구현 시 주의해야할 것 같은 부분(논문 본문 복붙)**<BR>
- input 을 만드는 것(segment token, mask, CLS, SEP 등.. **최소 1주 걸림..**)
- MLM 
- transformer 구조
- 모델이 복잡한건 아닌데 이것저것 디테일이 많아서 처음에 구조를 잘 짜놓으면 편할것 같다

## 🤫 논문과 다르게 구현한 부분
  
- 한국어 데이터
- optimizer : AdamW
- scheduler : 
- 문장이 길 때 max_seq_len을 자르는 부분 ? senB를 먼저 자르도록 했는데 논문에선 어떻게 자르는지 나와있진 않음
  
## 🤭 논문 구현하면서 배운 점 / 느낀 점
 
- BERT, transformer의 seq_len 차원은 정해져있을 필요가 없고 max_len으로 배치별로 패딩하는거다<br>
  -> 어차피 마지막 차원은 hid_dim이고 그걸로 연산을 함<br>
  -> seq_len 차원은 남아있음, transformer의 encoder의 ouput 차원은 [batch size, src len, hid dim]<br>
  -> max_seq_len은 OOM을 막기 위해 있는거다<br>
- NSP는 같은 문단에서도 바로 다음 문장이 아니면 0이다 
- [mask] 중에 10%는 그대로 남기는 이유 : [mask]로 선택된 토큰은 예측을 하는데, input이 [mask]이거나, random이거나 그대로 토큰일 때 맞추는 것
- bert에서 senA, senB의 단위는 일반적으로 문장임. finetune task에서 QA등을 할때에는 문단이나 단어가 들어갈 수 있음.
- 데이터가 클 때 리스트에 올리거나 하면 메모리 에러가 남..큰 데이터셋을 다룰 때는 기존의 방법이랑 다르게 해야함 -> linecache 사용해봄
- finetuning을 할 때, token embedding만 사용한다고 생각했었는데 그게 아닌 attention 값(batch_size, seq_len, hidden_dim)을 사용함!
- AdamW의 위력..initialize의 위력..
- tokenizers 사용법
- tokenizers 사용하여 chr-level NER input/output 만드는 코드!
