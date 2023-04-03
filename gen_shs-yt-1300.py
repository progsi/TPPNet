import pandas as pd
import argparse

def data_to_ID_list(data):
    return ['_'.join([str(e) for e in l]) for l in data[["set_id", "candidate_yt_id"]].values.tolist()]

def main(path: str):
    data = pd.read_csv(path, sep=';')
    data.head()
    
    with open('data/shs-yt-1300.txt', 'w') as f:
        for row in data_to_ID_list(data):
            f.write(row + '\n')
            
     

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='data/shs1300.csv')
    args = parser.parse_args()
    main(args.path) 

