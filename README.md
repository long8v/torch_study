# 🔥 torch_study 🔥

## environment
```
torch == 1.7.1
torchtext == 0.8.1
```

## Paper
🍟 [스터디 진행 중..](https://www.notion.so/kickoff-6634847c450741a68c1be736f102ecdd) 

🍕 `torch`, `torchtext`만 사용하여 패키지 형태로 논문 재현하기  

🍔 2021.01.06~2021.06.30까지 끝내기 목표

🌮 논문 구현 단계는 이와 같음

`논문 읽기` -> `reference code 읽기` -> `코드 짜기` -> `동일 조건 실험으로 성능이 재현되는지 확인하기`

|순번|paper|시작일자|상태|arxiv|notion|reference code|
|:--:|----|:-----:|:----:|:----:|:----:|:----:|
|01|[Convolutional Neural Networks for Sentence Classification](https://github.com/long8v/torch_study/tree/master/paper/01_CNN) (2014)|0106|Done|[paper](https://arxiv.org/abs/1408.5882)|[notion](https://www.notion.so/long8v/1-CNN-8982f7be7db94ae88e340cc23900693c)|[reference code](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/4%20-%20Convolutional%20Sentiment%20Analysis.ipynb)<br>[official code](https://github.com/yoonkim/CNN_sentence)|
|02|[Sequence to Sequence Learning with Neural Networks](https://github.com/long8v/torch_study/tree/master/paper/02_seq2seq) (2014)|0203|WIP|[paper](https://arxiv.org/abs/1409.3215)|[notion](https://www.notion.so/long8v/2-seq2seq-68f1ccd8f7c9451191334eae6f83486c)|[reference code1](https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb)<br>[reference code2](https://github.com/bentrevett/pytorch-seq2seq/blob/master/4%20-%20Packed%20Padded%20Sequences%2C%20Masking%2C%20Inference%20and%20BLEU.ipynb)|
|03|[Neural Machine Translation by Jointly Learning to Align and Translate](https://github.com/long8v/torch_study/tree/master/paper/03_attention) (2016)|0303|Done|[paper](https://arxiv.org/pdf/1409.0473.pdf)|[notion](https://www.notion.so/long8v/3-attention-22ac89a1000f49cba65aaa4a0a2ce9fa)|[reference code](https://github.com/bentrevett/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb)|
|04|[Attention is All You Need](https://github.com/long8v/torch_study/tree/master/paper/04_transformer) (2017)|0406|paper implementation|[paper](https://arxiv.org/abs/1706.03762)||[reference code](https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb)|




## *Tutorial(완료)* 
🍕 https://tutorials.pytorch.kr/ 노트북 다운 받아서 실행해보고 모르는 것 정리하고, 요약하고 연습문제 만들어서 풀기 

🍔 2020.09.30~2020.12.31까지 끝내기 목표 

|순번|튜토리얼|시작일자|상태|주요내용|링크|
|:--:|----|:---:|:----:|----|----|
|01|what is torch.nn?|1010|완료|`nn.Functional`,`nn.Module`,`nn.Linear`,`optim`|[tutorial](https://tutorials.pytorch.kr/beginner/nn_tutorial.html)|
|02|Tensorboard|1012|완료|`tensorboard`|[tutorial](https://tutorials.pytorch.kr/intermediate/tensorboard_tutorial.html)|
|03|이름분류|1013|완료|`nn.Linear`,`Dataset`,`DataLoader`|[tutorial](https://tutorials.pytorch.kr/intermediate/char_rnn_classification_tutorial.html)|
|04|이름생성|1019|완료|`nn.GRU`,`.to(device)`|[tutorial](https://tutorials.pytorch.kr/intermediate/char_rnn_generation_tutorial.html)|
|05|seq2seq 번역|1030|완료|`nn.Embedding`,`torch.save`,`torch.load`|[tutorial](https://tutorials.pytorch.kr/intermediate/seq2seq_translation_tutorial.html)|
|06|torchtext 분류|1023|완료|`torchtext`,`Field`,`nn.EmbeddingBag`|[tutorial](https://tutorials.pytorch.kr/beginner/text_sentiment_ngrams_tutorial.html)|
|07|torchtext 번역|1026|완료|`TabularDataset`,`BucketIterator`|[tutorial](https://tutorials.pytorch.kr/beginner/torchtext_translation_tutorial.html)|
|08|seq2seq 모델링|1022|완료|`nn.TransformerEncoder`|[tutorial](https://tutorials.pytorch.kr/beginner/transformer_tutorial.html)|


[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Flong8v%2F&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
