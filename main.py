from Data_Processer.Data_to_Feature import data2feature
from Data_Processer.Feature_to_PCG import feature2pcg
from Model_Trainer.Train_and_Test import train, eva

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="FastPack")
    parser.add_argument('--mode', type=str, required=True, help='Mode of operation: prepro, train, or eva')
    
    args = parser.parse_args()

    if args.mode == 'prepro':
        data2feature()
        feature2pcg()
    elif args.mode == 'train':
        train()
    elif args.mode == 'eva':
        eva()
    else:
        print("Please choose from 'prepro', 'train',  or  'eva'.")