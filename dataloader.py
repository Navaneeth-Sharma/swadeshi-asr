import pandas as pd
from datasets import Dataset
import json


train_data = pd.read_csv('train/transcription.txt',header = None)
train_data.columns = ["label"]

test_data = pd.read_csv('test/transcription1.txt',header = None)
test_data.columns = ["label"]

train_data[['audio_path','text']] = train_data["label"].str.split(" ", 1, expand=True)
test_data[['audio_path','text']] = test_data["label"].str.split(" ", 1, expand=True)

train_data = train_data[["audio_path","text"]]
test_data = test_data[["audio_path","text"]]


def add_file_path(text, train = True):
    if train:
        text = "train/audio/" + text + ".wav"
    else:
        text = "test/audio/" + text + ".wav"

    return text

# add file path
train_data['audio_path'] = train_data['audio_path'].map(lambda x: add_file_path(x))
test_data['audio_path'] = test_data['audio_path'].map(lambda x: add_file_path(x,train= False))


TRAIN_DATA = Dataset.from_pandas(train_data)
TEST_DATA = Dataset.from_pandas(test_data)


def extract_all_chars(batch):
    all_text = " ".join(batch["text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

vocab_train = train_data.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=train_data.column_names)
vocab_test = test_data.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=test_data.column_names)


vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))

vocab_dict = {v: k for k, v in enumerate(vocab_list)}


vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)


with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)
