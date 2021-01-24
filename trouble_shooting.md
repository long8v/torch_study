## 일반적인 상황

### RuntimeError: CUDA error: device-side assert triggered
device cpu로 바꾸고 디버깅 진행하기

### IndexError: Target 3226 is out of bounds.
target의 min, max 를 확인해본다 

### TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'torchtext.data.example.Example'>
TabularDataset은 torchtext의 Iterator와 연동되고, DataLoader랑 하려면 DataSet을 써야함


### expected scalar type long but found float 
target = target.long()     

The target should be a LongTensor using nn.CrossEntropyLoss (or nn.NLLLoss), since it is used to index the output logit (or log probability) for the current target class as shown in this formula 422 (note the indexing in x[class]).                                                     
### new() received an invalid combination of arguments - got (int, dtype=type), but expected one of: * (*, torch.device device) didn't match because some of the keywords were incorrect: dtype * (torch.Storage storage) * (Tensor other) * (tuple of ints size, *, torch.device device) * (object data, *, torch.device device)
dtype을 int로 주고싶은 경우
torch.Tensor(xx, dtype=torch.int64) 가 아니라 torch.Tensor(xx).long()와 같이 짜기

## 특수 상황

### ValueError: Expected target size (20, 11), got torch.Size([20, 10])
우리가 하는 것은 machine translation인데 이 경우 output의 길이는 target의 길이와 다를 수 있음
RNN은 한번에 한개씩 나와서 loss를 합쳐서 계산하지만 이 경우에는 어떻게 하지..?
-> seq2seq문제인데 encoder만 써서 분류 문제를 풀어서 그럼 seq2seq으로 접근해야함
optimizer.zero_grad() 이유


### forward() got an unexpected keyword argument 'tgt_mask'
TransformerDecoder에 TransformerEncoderLayer 인스턴스를 넣어서 발생 -> 정정
