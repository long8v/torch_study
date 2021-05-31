## environment
```
mlflow == 1.15.0
torch == 1.7.1
torchtext == 0.8.1
pytorch-lightning == 1.2.8
```


## result
### pretraining
`run_main.py`
![image](https://user-images.githubusercontent.com/46675408/120097868-c5db1f00-c16d-11eb-91fa-41763c01a640.png)
|train loss|train accuracy|
|---|---|
|0.343|0.55|


### finetuning
- data : KLUE
- task : topic classification

#### accuracy
![image](https://user-images.githubusercontent.com/46675408/120183336-82a2ae00-c24a-11eb-8937-3ce061567e93.png)
|train accuracy|valid accuracy|
|---|---|
|0.95|0.79|

#### fscore
![image](https://user-images.githubusercontent.com/46675408/120183572-d614fc00-c24a-11eb-9aa5-5a5069c7bf29.png)
|train f-score|valid f-score|
|---|---|
|0.9319|0.7639|

#### ablation study
. gamma vector added :

http://127.0.0.1:5000/#/compare-runs?runs=[%22b0b7ec4163064262bf532d9b4e7d1555%22,%228e87923752b14f9e8785880d16586fd1%22]&experiment=0

. chr vs token while finetuning :

![image](https://user-images.githubusercontent.com/46675408/120248286-7867ca80-c2b1-11eb-9688-7bdecb50654a.png)

캐릭터 단위가 오히려 높음 unk 토큰이 많아서 그런가?  
