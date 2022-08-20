from clean.master import MasterCleaner
from utils import json_reader, simple_reader, simple_writer


if __name__ == "__main__":
    config = json_reader("cleaner_config.json")

    cleaner = MasterCleaner(config)

    target_data = simple_reader("dataset/test_data.txt")
    corpus = list()
    for line in target_data:
        new_line = cleaner.cleaning(line)
        corpus.append(new_line)
    simple_writer("output/test_data_cleaned.txt", corpus)

    # faster approach
    """
    writer = codecs.open('output/test_data_cleaned.txt', 'w', encoding='utf-8')
    target_data = simple_reader('dataset/test_data.txt')
    corpus = list()
    for line in target_data:
        new_line = cleaner.cleaning(line)
        writer.write(new_line + '\n')
    writer.close()
    """
