
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset Builder')
    parser = argparse.ArgumentParser(description='Dataset Builder')
    parser.add_argument('-k', '--kfold', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=10)
    parser.add_argument('-p', '--path', type=str, default='/home/long8v')
    parser.add_argument('-w', '--w2v_path', type=str, default='/home/long8v/Downloads/GoogleNews-vectors-negative300.bin.gz')
    parser.add_argument('-n', '--n_filters', type=int, default=100)
    parser.add_argument('-f', '--filters_sizes', type=list, default=[3, 4, 5])
    parser.add_argument('-o', '--output_dim', type=int, default=2)
    parser.add_argument('-d', '--dropout', type=float, default=0.5) 
    parser.add_argument('-epochs', '--n_epochs', type=int, default=10)   
    args = parser.parse_args()    
    print(args)
    print(args.batch_size)