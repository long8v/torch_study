import logging
import torch
from datetime import datetime
from sacrebleu import corpus_bleu, sentence_bleu


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def logging_model(model):
    logging.info(model)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def logging_count_parameters(model):
    print(f'The model has {count_parameters(model):,} trainable parameters')
    logging.info(f'The model has {count_parameters(model):,} trainable parameters')

def get_today():
    return datetime.today().strftime("%Y/%m/%d%H%M")

def logging_time():
    logging.info(get_today())

def read_json(path):
    with open(path, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
    return json_data

            
def get_speical_token(field):
    def get_stoi(idx):
        return field.vocab.stoi[idx]
    return [get_stoi(field.pad_token), get_stoi(field.unk_token), 
            get_stoi(field.eos_token)]

def get_itos_str(tokens, field):
    ignore_idx = get_speical_token(field)
    return ' '.join([field.vocab.itos[token] for token in tokens
                    if token not in ignore_idx])

def get_itos_batch(tokens_batch, field):
    return [get_itos_str(batch, field) for batch in tokens_batch]

def get_bleu_score(output, trg, trg_field):
    with torch.no_grad():
        output_token = output.argmax(-1)
    # 문장 별로 해야돼서 permute 해야 함
    output_token = output_token.permute(1, 0)
    trg = trg.permute(1, 0)
    system = get_itos_batch(output_token, trg_field)
    refs = get_itos_batch(trg, trg_field)
    bleu = corpus_bleu(system, [refs], force=True).score
    return bleu