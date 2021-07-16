## 🤗 Result
### Environment
```
mlflow == 1.15.0
torch == 1.7.1
torchtext == 0.8.1
pytorch-lightning == 1.2.8
tokenizers == 0.9.3
pytorch-crf == 0.7.2
dill == 0.3.4
```

### pretraining
**MLM**
![image](https://user-images.githubusercontent.com/46675408/124102640-81f47600-da9b-11eb-86e8-18f7897fae89.png)
![image](https://user-images.githubusercontent.com/46675408/124102899-bb2ce600-da9b-11eb-814f-30b2761b0f5c.png)


**NSP**
![image](https://user-images.githubusercontent.com/46675408/124102824-a7817f80-da9b-11eb-8217-a6dff6d797dd.png)
![image](https://user-images.githubusercontent.com/46675408/124103027-d992e180-da9b-11eb-8246-83efdb01650d.png)


## 🤔 Paper review
**1) PPT 한 장 분량으로 자유롭게 논문 정리 뒤 이미지로 첨부**


**2) (슬랙으로 이미 토론을 했지만 그래도) 이해가 안 가는 부분, 이해가 안 가는 이유(논문 본문 복붙)**<BR>
. MLM을 정확히 어떻게 하는건지 모르겠음 mask 토큰 넣은채로 인풋에 넣은 뒤에 mask 토큰에 대해서만 predict 하게 하려면 어떻게 구현해야하지?<br>
. [MASK] 토큰을 15% 선정하고 80%은 바꾸고 10%은 치환하고 **10%는 그냥 둔거..** 10% 그냥 두는게 의미가 뭐지 걍 둔건가..<BR>

original :    i go to shcool.<BR>
inference :    i [mask] to **[mask].** 15%<BR>
input :        **i [mask] to school.**<BR>
. 선정된 [MASK]들의 prediction을 할 때 그냥 둔 토큰의 임베딩이 들어감

**3) 재밌었던 부분**<BR>
. MLM - transformer아이디어 간단하고 직관적임. <BR>
. BERTbase가 GPT랑 파라미터 개수 맞춰서 나온 것인 것..ㅎㅎ<BR>
. ELMo 얘기 나오는데 왠지 고향친구 만난듯한 반가움..<BR>
. **feature base**랑 **fine-tuning**이랑 깔끔하게 나눠준거. 항상 fine-tuning 얘기하면 논란이 되는 부분이었음.<BR>
. 그리고 또 feature base로 문제를 풀어주심<BR>
. 데이터를 문장 단위로 잘라서 SHUFFLE한 데이터셋보다는 DOCUMENT 단위의 데이터셋을 써야지 긴 문맥을 뽑아낼수 있어서 좋다고 함.<BR>
  
**4) 논문 구현 시 주의해야할 것 같은 부분(논문 본문 복붙)**<BR>
- input 을 만드는 것(segment token, mask, CLS, SEP 등.. **최소 1주 걸림..**)
- MLM 
- transformer 구조..난 까먹음..
- 모델이 복잡한건 아닌데 이것저것 디테일이 많아서 처음에 구조를 잘 짜놓으면 편할것 같다
- 허깅페이스 트랜스포머 한 주 보고 가자!

## 🤫 논문과 다르게 구현한 부분

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
