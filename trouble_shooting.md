### RuntimeError: CUDA error: device-side assert triggered
device cpu로 바꾸고 진행하기

### IndexError: Target 3226 is out of bounds.
target의 min, max 를 확인해본다 

### TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'torchtext.data.example.Example'>
TabularDataset은 BucketIterator랑 쌍으로 사용돼서 그런듯?


### forward() got an unexpected keyword argument 'tgt_mask'
TransformerDecoder에 TransformerEncoderLayer 인스턴스를 넣어서 발생 -> 정정
