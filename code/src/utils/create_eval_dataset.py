import pandas as pd
import os
from argparse import ArgumentParser
import numpy as np
from itertools import combinations
from termcolor import colored
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#       TO RUN:      python create_eval_dataset.py -p <path_to_tabular_data> -o <output_path>

#computes the difference in days
def to_days_difference(ses):
    out = []
    for i in range(len(ses)-1):
        out.append(int((ses[i+1] - ses[i])/ np.timedelta64(1, 'D')))

    out.append(0)
    return out


# create all posslible sequences
def create_sequences(df, seq_id):
    sequences = []
    counter = seq_id
    for i in range(2, len(df) + 1):
        for c in combinations([d for d in range(len(df))], i):
            idxs = [j for j in c]
            sequence = pd.DataFrame({"SEQ_ID": [seq_id + counter for x in idxs]})
            sequence = pd.concat([sequence, df.loc[idxs].reset_index(drop=True)], axis=1)
            sequences.append(sequence)
            counter += 1

    return sequences, counter

#create a dictionary for all subjects with their respective sequences
def create_subject_sequences(df):
    # unique subject list
    subjects = list(df.ID.unique())

    dfs = []

    ages = []
    days = []
    sexes = []

    seq_id = 0

    for sub in subjects:
        temp = df[df.ID == sub].sort_values(by="ses").reset_index(drop=True)
        if len(temp) > 1:
            df_sub = pd.DataFrame(temp.ID)
            df_sub["sex"] = [1 if x == "F" else 0 for x in temp.sex]
            df_sub["age"] = temp.age
            df_sub["ses"] = temp.ses
            # print(df_sub.head())
            ages += list(temp.age)
            sexes.append(df_sub.sex.unique()[0])
            days += to_days_difference(list(df_sub.ses))[:-1]
            schaefer_rois = temp.loc[:, 'SUVR.Schaefer200.ROI.idx.1':'SUVR.Schaefer200.ROI.idx.200']
            dx = temp.loc[:, "DX"]
            df_sub = pd.concat([df_sub, schaefer_rois, dx], axis=1)
            sub_sequences, seq_id = create_sequences(df_sub, seq_id)
            for seq in sub_sequences:
                seq["days_to_next"] = to_days_difference(list(seq.ses))

            dfs += sub_sequences

    return dfs, ages, days, sexes



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-p", "--dataset_path", dest="dataset_path",
                        help="tabular dataset path", required=True)
    parser.add_argument("-o", "--output_path", dest="output_path", required=True)

    args = parser.parse_args()

    #supress warning
    pd.options.mode.chained_assignment = None

    #path for tabular data
    dataset_path = args.dataset_path

    #output path where to create the dataset
    output_path = args.output_path

    print("Computing ...")
    #read dataset with pandas
    df = pd.read_csv(dataset_path)

    #extract session list
    sessions = list(df.ses)

    print("Sanity checking ...")
    # sanity check that every session name starts with ses
    sanity_session = True
    for s in sessions:
        if not s.startswith("ses-"):
            sanity_session = False
            break

    assert sanity_session
    print(colored("Sanity check passed!", "green"))
    # remove ses prefix
    sessions = [s[4:] for s in sessions]
    df.ses = sessions


    # convert column to pandas datetime to allow sorting
    df.ses = pd.to_datetime(df.ses)

    print("Generating subject sequences ...")
    #create the sequences for all subjects
    dfs, ages, days, sexes = create_subject_sequences(df)
    print(colored("Sequences successfully generated!", "green"))

    print("Scaling features ...")
    age_scaler = MinMaxScaler((0, 1))
    age_scaler.fit(np.array(ages)[:, np.newaxis])

    sex_scaler = StandardScaler()
    sex_scaler.fit(np.array(sexes)[:, np.newaxis])

    days_scaler = MinMaxScaler((0, 1))
    days_scaler.fit(np.array(days)[:, np.newaxis])

    for seq in dfs:
        seq["sex"] = sex_scaler.transform(np.array(seq.sex)[:, np.newaxis]).flatten()
        seq["age"] = age_scaler.transform(np.array(seq.age)[:, np.newaxis]).flatten()
        sub_days = days_scaler.transform(np.array(seq.days_to_next)[:, np.newaxis]).flatten()
        sub_days[-1] = 0
        seq["days_to_next"] = sub_days

    out = pd.concat([x for x in dfs]).reset_index(drop=True)

    print(colored("Scaling complete!", "green"))

    print("Creating dir and saving ...")
    #create dataset path
    dataset_root = os.path.join(output_path, "tau_progression_sequences_eval.csv")

    out.to_csv(dataset_root, index=False)
    print(colored("Saved!", "green"))
    print(colored("all sequences created successfully!", "green"))
    print("created dataset path: " + os.path.abspath(dataset_root))



