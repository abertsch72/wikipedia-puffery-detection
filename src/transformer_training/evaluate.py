import os
from typing import Text
import csv

import transformers
from sklearn.metrics import precision_recall_fscore_support

from src import transformer_training as models
from src.transformer_training import get_all_data


MODEL_PATH = "saved_models/testmodel.11-1.96-f1at0.72727.hdf5"
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
    model = models.from_transformer(transformer=transformer, n_outputs=1)
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

nonpeacock = os.path.join("clean-nonpeacockterms.txt")
peacock = os.path.join("clean-peacockterms.txt")

test_model(MODEL_PATH, peacock, nonpeacock, "roberta-base", 6)