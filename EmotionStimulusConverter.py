import argparse
import os
import pandas as pd
import numpy as np

# Takes input and output directories as arguments
parser=argparse.ArgumentParser()
parser.add_argument('--input', default=".", help='The file path of the unzipped Emotion Stimulus dataset')
parser.add_argument('--output', default="./data", help='The file path of the output dataset')

args = parser.parse_args()
INPUT_PATH = args.input
OUTPUT_PATH = args.output

def create_emo_stim_df(filename):
    f = open(INPUT_PATH +"/"+ filename, "r")

    fl =f.readlines()

    emotions = ["happy", "sad", "surprise", "disgust", "anger", "fear", "shame"]

    texts = []
    labels = []

    def tag_remover_and_labeller(line):
        for emotion in emotions:
            emotion_tag = "<"+emotion+">"
            emotion_closing_tag = "<\\"+emotion+">"
            if emotion_tag in line:
                label = emotion

            line = line.replace(emotion_tag, "").replace(emotion_closing_tag, "")

        line = line.replace("<cause>", "").replace("<\\cause>", "")

        return({"text": line, "emotion": label})

    return(pd.DataFrame(list(pd.Series(fl).apply(tag_remover_and_labeller))))

emo_stim = create_emo_stim_df("Emotion Cause.txt").append(create_emo_stim_df("No Cause.txt")).reset_index(drop=True)

fraction = 0.2

np.random.seed(seed=42)

test_indices = np.random.choice(emo_stim.index, size=int(round(fraction*emo_stim.shape[0])), replace=False)
train_indices = emo_stim.index.difference(test_indices)
dev_indices = np.random.choice(train_indices, size=int(round(fraction*len(train_indices))), replace=False)
train_indices = train_indices.difference(dev_indices)

emo_stim_train = emo_stim.loc[train_indices,:]
emo_stim_dev = emo_stim.loc[dev_indices,:]
emo_stim_test = emo_stim.loc[test_indices,:]

emo_stim_train.reset_index(drop=True).to_csv(OUTPUT_PATH+"/train.tsv", sep='\t', encoding="utf-8")
emo_stim_dev.reset_index(drop=True).to_csv(OUTPUT_PATH+"/dev.tsv", sep='\t', encoding="utf-8")
emo_stim_test.reset_index(drop=True).to_csv(OUTPUT_PATH+"/test.tsv", sep='\t', encoding="utf-8")
