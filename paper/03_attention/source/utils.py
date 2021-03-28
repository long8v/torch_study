import logging
from datetime import datetime


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def logging_model(model):
    loggin.info(model)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def logging_count_parameters(model):
    print(f'The model has {count_parameters(model):,} trainable parameters')
    logging.info(f'The model has {count_parameters(model):,} trainable parameters')

def get_today():
    return datetime.today().strftime("%Y/%m/%d%H%M")

def logging_time():
    logging.info(get_today())