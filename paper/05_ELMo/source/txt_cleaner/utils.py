import codecs
import json


def json_reader(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print('size {} dictionary is read from {}'.format(len(data), path))
    return data


def simple_reader(path):
    corpus = list()
    reader = codecs.open(path, 'r', encoding='utf-8')
    for line in reader:
        corpus.append(line.strip())
    reader.close()
    print('{} lines in read from {}'.format(len(corpus), path))
    return corpus


def simple_writer(path, target):
    writer = codecs.open(path, 'w', encoding='utf-8')
    for line in target:
        writer.write(line.strip() + '\n')
    writer.close()
    print('{} lines is written in {}'.format(len(target), path))