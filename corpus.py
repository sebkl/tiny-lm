import numpy as np
import tensorflow as tf
import logging


DEFAULT_BATCH_SIZE = 16
DEFAULT_BLOCK_SIZE = 32


class CharacterEncoder:
    """Encode ASCII strings into char set indices (tokens)."""

    def __init__(self, text):
        self._charset = sorted(list(set(text)))
        self._reverse_map = dict()
        self._vocab_size = len(self._charset)

        for i, c in enumerate(self._charset):
            self._reverse_map[c] = i

    def encode(self, text):
        """Encode a string into a list of indices.

        Parameters
        text (str): The input sequence to encode.

        Returns:
        [int]: A list if 32 bit integer indices (tokens)."""

        return [self._reverse_map[c] for c in text]

    def decode(self, tokens):
        """Decode a list of 32 bit integer indices (tokens)
        to a ASCII string.

        Parameters:
        tokens ([int]): A list if 32 bit integer indices (e.g. tokens).

        Returns:
        str: Decoded ASCII string."""

        return "".join([self._charset[i] for i in tokens])

    @property
    def vocab_size(self):
        return self._vocab_size


class TextCorpus:

    def __init__(self, text, encoder, validation_ratio=0.1):
        self._plain_data = text
        self._encoder = encoder
        super().__init__()

        self._logger = tf.get_logger()
        self._logger.setLevel(logging.INFO)
        text_tensor = tf.constant(
            self._encoder.encode(self._plain_data), dtype=tf.int32
        )
        split_threshold = int(len(text_tensor) * (1.0 - validation_ratio))
        self._train_data = text_tensor[:split_threshold]
        self._validation_data = text_tensor[split_threshold:]
        self._logger.info(
            "Train data size: %d tokens, validation data size: %d tokens."
            % (len(self._train_data), len(self._validation_data))
        )

    @property
    def encoder(self):
        return self._encoder

    def get_batch(
        self,
        block_size=DEFAULT_BLOCK_SIZE,
        batch_size=DEFAULT_BATCH_SIZE,
        validation=False,
    ):
        """Extracts a random batch from the training dataset.

        Parameters:
        block_size (int): Overwrite default block size
        batch_size (int): Overwrite default batch size
        validation (bool): From validation set

        Returns:
        int,int: data input batch, expected value batch.
        """
        d = self._train_data
        if validation:
            d = self._validation_data

        # Pick a random start index based on given batch and block size.
        ix = tf.random.uniform(
            shape=(batch_size,),
            minval=0,
            maxval=(len(d) - block_size),
            dtype=tf.int32,
        )

        # Construct data and prediction batch.
        x = tf.stack([d[i : i + block_size] for i in ix])
        y = tf.stack([d[i + 1 : i + block_size + 1] for i in ix])
        return x, y

    def _gen(self, size=None, batch_size=None, block_size=None):
        """Generate data batches indefinitely based on a fixed size text."""

        while True:
            x, y = self.get_batch(batch_size=batch_size, block_size=block_size)
            yield x, y

    def as_dataset(
        self, size=None, block_size=DEFAULT_BLOCK_SIZE, batch_size=DEFAULT_BATCH_SIZE
    ):
        """Return corpus as generator based tensorflow dataset."""

        def gen():
            return self._gen(size, batch_size=batch_size, block_size=block_size)

        return tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=(batch_size, block_size), dtype=tf.int32),
                tf.TensorSpec(shape=(batch_size, block_size), dtype=tf.int32),
            ),
        )
