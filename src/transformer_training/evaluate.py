import os
from typing import Text
import csv

import transformers
from sklearn.metrics import precision_recall_fscore_support

from src import transformer_training as models
import tensorflow as tf
import transformers


def from_transformer(transformer: transformers.TFPreTrainedModel,
                     n_outputs: int) -> tf.keras.Model:

    # Define inputs (token_ids, mask_ids, segment_ids)
    token_inputs = tf.keras.Input(shape=(None,), name='word_inputs', dtype='int32')
    mask_inputs = tf.keras.Input(shape=(None,), name='mask_inputs', dtype='int32')
    segment_inputs = tf.keras.Input(shape=(None,), name='segment_inputs', dtype='int32')

    # get contextualized token encodings from transformer
    token_encodings = transformer([token_inputs, mask_inputs, segment_inputs])[0]

    # get a sentence encoding from the token encodings
    sentence_encoding = tf.keras.layers.GlobalMaxPooling1D()(token_encodings)

    # Final output layer
    outputs = tf.keras.layers.Dense(n_outputs, activation='sigmoid', name='outputs')(sentence_encoding)

    # Define model
    return tf.keras.Model(inputs=[token_inputs, mask_inputs, segment_inputs], outputs=[outputs])
#from src.transformer_training import data.get_all_data

import numpy as np
import pandas as pd
import transformers
from typing import Mapping, Sequence, Text, Union
import random
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



MODEL_PATH = "testmodel.03-0.23-f1at0.96296.hdf5"
print(MODEL_PATH)
def test_model(model_path: Text,
         positive_examples_file: Text,
         negative_examples_file: Text,
         pretrained_model_name: Text,
         batch_size: int):

    width = 40
    headers = ["precision", "recall", "f1-score", "support"]
    header_fmt = f'{{:<{width}s}} ' + ' {:>9}' * 4
    row_fmt = f'{{:<{width}s}} ' + ' {:>9.3f}' * 3 + ' {:>9}'

    # load the tokenizer model
    tokenizer_for = transformers.AutoTokenizer.from_pretrained
    tokenizer = tokenizer_for(pretrained_model_name)

    # load the pre-trained transformer model
    model_for = transformers.TFAutoModel.from_pretrained
    transformer = model_for(pretrained_model_name)

    test_data_rows = {positive_examples_file: [], negative_examples_file: []}

    # load the fine-tuned transformer model
    model = from_transformer(transformer=transformer, n_outputs=1)
    model.load_weights(model_path) #.expect_partial()
    raw_x, data_x, data_y = get_all_data(tokenizer, negative_examples_file, positive_examples_file)

    # predict on the test data
    y_pred_scores = model.predict(data_x, batch_size=batch_size)
    y_pred = (y_pred_scores >= 0.5).astype(int).ravel()
    out = [["sentence", "label", "prediction"]]
    for i in range(len(raw_x)):
        out.append([raw_x[i], data_y[i], y_pred[i]])

    with open("predictions.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerows(out)

    # evaluate predictions
    stats_arrays = precision_recall_fscore_support(
        data_y, y_pred, labels=[1])
    stats = [a.item() for a in stats_arrays]
    row = [model_path] + stats
    print(stats)


        # if requested, print detailed results for this model
    # print results for all models on all datasets
    for data_path, rows in test_data_rows.items():
        print(header_fmt.format(data_path, *headers))
        for row in rows:
            print(row)
        print()

nonpeacock = os.path.join("..", "..", "..", "data", "NEW-clean-nonpeacock-ir.txt")
peacock = os.path.join("..", "..", "..", "data","NEW-clean-peacockterms.txt")

test_model(MODEL_PATH, peacock, nonpeacock, "roberta-base", 8)