import logging
import argparse

logging.basicConfig(filename='example.log', level=logging.DEBUG)
logging.debug('This message should go to the log file')
logging.info('So should this')
logging.warning('And this, too')
logging.error('And non-ASCII stuff, too, like Øresund and Malmö')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset Builder')
    parser.add_argument('-k', '--kfold', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=50)
    parser.add_argument('-p', '--path', type=str, default='/home/long8v')
    parser.add_argument('-w', '--w2v_path', type=str, default='/home/long8v/torch_study/data/GoogleNews-vectors-negative300.bin.gz')
    parser.add_argument('-n', '--n_filters', type=int, default=100)
    parser.add_argument('-f', '--filters_sizes', type=list, default=[3, 4, 5])
    parser.add_argument('-o', '--output_dim', type=int, default=2)
    parser.add_argument('-d', '--dropout', type=float, default=0.5) 
    parser.add_argument('-epochs', '--n_epochs', type=int, default=20)   
    args = parser.parse_args()
    print(args)
    # main()tut4-model