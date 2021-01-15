### RuntimeError: CUDA error: device-side assert triggered
device cpu로 바꾸고 진행하기

### IndexError: Target 3226 is out of bounds.
target의 min, max 를 확인해본다 

### TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'torchtext.data.example.Example'>
TabularDataset은 BucketIterator랑 쌍으로 사용돼서 그런듯?


### forward() got an unexpected keyword argument 'tgt_mask'
TransformerDecoder에 TransformerEncoderLayer 인스턴스를 넣어서 발생 -> 정정

### ValueError: Expected target size (20, 11), got torch.Size([20, 10])
우리가 하는 것은 machine translation인데 이 경우 output의 길이는 target의 길이와 다를 수 있음
RNN은 한번에 한개씩 나와서 loss를 합쳐서 계산하지만 이 경우에는 어떻게 하지..?
-> seq2seq문제인데 encoder만 써서 분류 문제를 풀어서 그럼 seq2seq으로 접근해야함
optimizer.zero_grad() 이유
