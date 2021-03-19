import numpy as np
import pandas as pd
import transformers
from typing import Mapping, Sequence, Text, Union
import random


def get_data(tokenizer: transformers.PreTrainedTokenizer, nonpeacock_filename, peacock_filename) -> (
        np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    df = [[line.strip(), 1] for line in open(peacock_filename).readlines()]
    df.extend([[line.strip(), 0] for line in open(nonpeacock_filename).readlines()])
    random.shuffle(df)
    df = np.array(df)
    split_point = int(len(df) * 0.9)
    train = df[:split_point]
    train_x = from_tokenizer(tokenizer, [data[0] for data in train])
    train_y = np.array([int(data[1]) for data in train])
    dev = df[split_point:]
    dev_x = from_tokenizer(tokenizer, [data[0] for data in dev])
    dev_y = np.array([int(data[1]) for data in dev])
    return train_x, train_y, dev_x, dev_y


def get_all_data(tokenizer: transformers.PreTrainedTokenizer, nonpeacock_filename, peacock_filename) -> (
        np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    df = [[line.strip(), 1] for line in open(peacock_filename).readlines()]
    df.extend([[line.strip(), 0] for line in open(nonpeacock_filename).readlines()])
    random.shuffle(df)
    df = np.array(df)
    x_raw = [data[0] for data in df]
    x = from_tokenizer(tokenizer, [data[0] for data in df])
    y = np.array([int(data[1]) for data in df])

    return x_raw, x, y

# converts texts into input matrices required by transformers
def from_tokenizer(tokenizer: transformers.PreTrainedTokenizer,
                   texts: Sequence[Text],
                   pad_token: int = 0) -> Mapping[Text, np.ndarray]:
    rows = [tokenizer.encode(text,
                             add_special_tokens=True,
                             max_length=tokenizer.model_max_length,
                             truncation=True)
            for text in texts]
    shape = (len(rows), max(len(row) for row in rows))
    token_ids = np.full(shape=shape, fill_value=pad_token)
    is_token = np.zeros(shape=shape)
    for i, row in enumerate(rows):
        token_ids[i, :len(row)] = row
        is_token[i, :len(row)] = 1
    return dict(
        word_inputs=token_ids,
        mask_inputs=is_token,
        segment_inputs=np.zeros(shape=shape))


def read_csv(
        data_path: Text,
        label_col: Text,
        n_rows: Union[int, None] = None) -> pd.DataFrame:
    df = pd.read_csv(data_path, nrows=n_rows)
    cols = {c.lower().replace(" ", ""): c for c in df.columns}
    [y_col] = [cols[c] for c in cols if label_col in c]
    [x_col] = [cols[c] for c in cols if "text" in c or "tweet" in c]
    df = df[[x_col, y_col]].dropna()
    if pd.api.types.is_string_dtype(df[y_col]):
        df[y_col] = pd.to_numeric(df[y_col].replace({"o": "0"}))
    return df.rename(columns={x_col: "text", y_col: label_col})


def df_to_xy(
        df: pd.DataFrame,
        tokenizer: transformers.PreTrainedTokenizer,
        label_col: Text) -> (np.ndarray, np.ndarray):
    x = from_tokenizer(tokenizer, df["text"])
    y = df[label_col].values
    return x, y


"""
so basically:
create np array of y vals, np array of x vals
run array of x vals through from_tokenizer
boom done!
"""


def read_csvs_to_xy(
        data_paths: Sequence[Text],
        tokenizer: transformers.PreTrainedTokenizer,
        label_col: Text,
        n_rows: Union[int, None] = None) -> (np.ndarray, np.ndarray):
    dfs = [read_csv(p, label_col=label_col, n_rows=n_rows) for p in data_paths]
    df = pd.concat(dfs)
    return df_to_xy(df, tokenizer, label_col=label_col)
