"""
https://github.com/clulab/incivility
"""
import transformers
import tensorflow as tf
import tensorflow_addons as tfa
from typing import Sequence, Text
import os

from transformer_training import models, data

model = "roberta-base"
model_path = os.path.join("..", "saved_models", "testmodel.{epoch:02d}-{val_loss:.2f}.hdf5")
nonpeacock = os.path.join("..", "NEW-clean-nonpeacock-ir.txt")
peacock = os.path.join("..", "NEW-clean-peacockterms.txt")
learning_rate = 0.00001 #0.00001
batch_size = 5 # should be okay to be small -- 6, 8 was best so far
n_epochs = 20 # might be quite small

"""
mess around with GPU libraries
"""


def train(model_path: Text,
          positive_examples_file: Text,
          negative_examples_file: Text,
          pretrained_model_name: Text,
          learning_rate: float,
          batch_size: int,
          n_epochs: int,
          use_gpu: bool = False):
    tokenizer_for = transformers.AutoTokenizer.from_pretrained
    tokenizer = tokenizer_for(pretrained_model_name)
    train_x, train_y, dev_x, dev_y = data.get_data(tokenizer, negative_examples_file, positive_examples_file)

    """
    # set class weight inversely proportional to class counts
    counts = np.bincount(train_y)
    class_weight = dict(enumerate(counts.max() / counts))
    """

    # determine optimizer
    optimizer_kwargs = dict(
        learning_rate=learning_rate, epsilon=1e-08, clipnorm=1.0)
    optimizer_class = tf.optimizers.Adam

    model_for = transformers.TFAutoModel.from_pretrained
    model = models.from_transformer(
        transformer=model_for(pretrained_model_name),
        n_outputs=1)
    model.compile(
        optimizer=optimizer_class(**optimizer_kwargs),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tfa.metrics.F1Score(num_classes=1, threshold=0.5),
        ])

    model.fit(x=train_x, y=train_y,
              validation_data=(dev_x, dev_y),
              epochs=n_epochs,
              batch_size=batch_size,
              # class_weight=class_weight,
              callbacks=tf.keras.callbacks.ModelCheckpoint(
                  filepath=model_path,
                  monitor="val_f1_score",
                  mode="max",
                  verbose=1,
                  save_weights_only=True,
                  save_best_only=True))

train(model_path, peacock, nonpeacock, model, learning_rate, batch_size, n_epochs)

