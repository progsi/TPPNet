import pandas as pd
import argparse

def data_to_ID_list(data):
    return [str(l[0]) + '_' + l[2] + '-' + str(round(l[1], 0)) 
            for l in data[["set_id", "nlabel", "candidate_yt_id"]].values.tolist()]

def main(path: str):
    """
    Transform the SHS-YT-1300 dataset into a textfile as input for TPPNet.
    Each set_id of the initial sets is mapped to 
    positives --> setID1_ytID 
    or negatives --> setID0_ytID 
    """
    data = pd.read_csv(path, sep=';')
    
    with open('data/shs-yt-1300-ext.txt', 'w') as f:
        for row in data_to_ID_list(data):
            f.write(row + '\n')
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='data/shs1300_ext.csv')
    args = parser.parse_args()
    main(args.path) 

