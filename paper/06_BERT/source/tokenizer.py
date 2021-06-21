import mecab
from utils import *
from glob import glob
import sys
sys.path.append('/home/long8v/torch_study/paper/05_ELMo/source')
from txt_cleaner.clean.master import MasterCleaner
from nltk.tokenize import sent_tokenize
from tokenizers.implementations import BertWordPieceTokenizer
import random

'''
1) 데이터 전처리
    - 문장으로 자르기
    - cleaning
    - 한 문서가 끝나면 <EOD> 붙이기
    - 문장을 mecab으로 자르기
    - ## 붙이기
2) tokenizers 학습
    - tokenizers 학습하기
    - tokenizers에 special tokens [SEP], [CLS] 추가
    - tokenizers 저장
'''

class prepare_BERT:
    '''
    corpus_name : list 
    
    '''
    def __init__(self, corpus_path, save_path):
        files = glob(f'{corpus_path}/*.p')
        print(files)
        self.pos = mecab.MeCab()
        self.cleaner = MasterCleaner({'minimum_space_count': 5})
        print('cleaning and pos tagging corpus..')
        f_mecab = open(f'{save_path}/bert_mecab.txt', 'w')
        f = open(f'{save_path}/bert.txt', 'w')
        f_valid = open(f'{save_path}/bert_valid.txt', 'w')
        for file in files:
            print(file)
            corpus = read_pickle(file)
            for paragraph in corpus:
                for sentence in sent_tokenize(paragraph):
                    sentence = self.cleaner.cleaning(sentence)
                    if sentence.strip():
                        if random.random() < 0.1:
                            f_valid.write(f'{sentence}\n')
                        else:
                            f.write(f'{sentence.strip()}\n')
                            sentence = self.get_morphs(sentence.strip())
                            f_mecab.write(f'{sentence}\n')

                f.write('<EOD>\n')
                f_valid.write('<EOD>\n')
        f.close()
        f_mecab.close()
        f_valid.close()
        files = glob(f'{corpus_path}/*_mecab.txt')
        self.tokenizer = BertWordPieceTokenizer(wordpieces_prefix='##', strip_accents=False)
        print('train tokenizer..')
        self.tokenizer.train(files=files,
                       min_frequency=5)
        self.tokenizer.add_special_tokens(['[SEP]', '[CLS]', '[MASK]'])
        print('save tokenizer model!')
        self.tokenizer.save(f'{save_path}/vocab.json')     

    
    def get_morphs(self, sentence):
        morph_sentence = ''
        original_sentence = sentence
        for word in self.pos.morphs(sentence):
            if morph_sentence:
                if original_sentence.startswith(' '):
                    morph_sentence += f' {word}'
                    original_sentence = original_sentence.strip() # 띄어쓰기 두 개 이상 있는 경우 
                else:
                    morph_sentence += f'##{word}'
                    original_sentence = original_sentence.strip() 
            else:
                morph_sentence += f'{word}'
            original_sentence = original_sentence[len(word):]
        return morph_sentence
    

if __name__ == '__main__':
    pb = prepare_BERT('/home/long8v/torch_study/paper/file/bert', '/home/long8v/torch_study/paper/file/bert')
    print(pb.tokenizer.get_vocab_size())
    print(pb.tokenizer.encode('안녕하쎼요').tokens)
    print(pb.tokenizer.encode('반갑습니다 저는 그런 생각을 했습니다.').tokens)