import tensorflow as tf


class Params:
    """Hyper parameters used to train the model.

    Defaults are picked to train a model getting somewhat reasonable results on a
    '24 mac book air withing 15 to 30 minutes."""

    # Embedding dimension size for each token.
    embedding_dim = 128

    # Number of transformer blocks.
    num_blocks = 4

    # Number of self-attention heads per block.
    num_heads = 4

    # Number of tokens per training sequence.
    block_size = 256

    # Size of a batch representing the number of text sequences per batch.
    batch_size = 32

    # Learning rate - https://en.wikipedia.org/wiki/Learning_rate
    learning_rate = 5e-4

    # Dropout rate - https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
    dropout_rate = 0.2

    # Iterations to run.
    epochs = 200

    # Training steps per iteration.
    steps_per_epoch = 20

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class BatchLoss(tf.keras.losses.Loss):
    """Compute loss over an set of batches."""

    def call(self, y_true, y_pred):
        """Compute batch loss.

        Parameters:
        y_true (b,t,c): Data batches.
        y_pred (b,t,v): Predicted batches.

        Returns:
        The loss value as float."""
        b, t, c = y_pred.shape
        y_pred_rs = tf.reshape(y_pred, [b * t, c])
        y_true_rs = tf.reshape(y_true, [b * t])
        return tf.keras.losses.sparse_categorical_crossentropy(
            y_true_rs, y_pred_rs, from_logits=True
        )


class HeadLayer(tf.keras.layers.Layer):
    """Attention head modeled as layer."""

    def __init__(self, embedding_dim, head_size, dropout_rate=0.1):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._head_size = head_size
        self._dropout_rate = dropout_rate

    def build(self, input_shape):
        block_size = input_shape[1]
        self._query = tf.keras.layers.Dense(self._head_size, use_bias=False)
        self._key = tf.keras.layers.Dense(self._head_size, use_bias=False)
        self._value = tf.keras.layers.Dense(self._head_size, use_bias=False)
        self._softmax = tf.keras.layers.Softmax(axis=-1)
        self._dropout = tf.keras.layers.Dropout(self._dropout_rate)

        # Create mask for past tokens to make sure only previous tokens influence
        # generated future tokens.
        self._mask = tf.linalg.band_part(tf.ones((block_size, block_size)), -1, 0) == 0

    def call(self, x):
        _, block_size, embedding_dim = x.shape
        k = self._key(x)
        q = self._query(x)
        k_t = tf.transpose(k, perm=[0, 2, 1])
        w = tf.matmul(q, k_t)
        w = w * (embedding_dim ** (-0.5))
        w = tf.where(self._mask[:block_size, :block_size], x=float("-inf"), y=w)
        w = self._softmax(w)
        w = self._dropout(w)
        v = self._value(x)
        return tf.matmul(w, v)


class MultiHeadLayer(tf.keras.layers.Layer):
    """Multi-head layer for self-attention.

    It consists of a configurable amount of self-attention heads."""

    def __init__(self, embedding_dim, num_heads, dropout_rate=0.1):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._head_size = embedding_dim // num_heads
        self._num_heads = num_heads
        self._concat = tf.keras.layers.Concatenate(axis=-1)
        self._dropout_rate = dropout_rate

    def build(self, input_shape):
        B, T, C = input_shape
        self._heads = [
            HeadLayer(
                self._embedding_dim, self._head_size, dropout_rate=self._dropout_rate
            )
            for _ in range(self._num_heads)
        ]
        self._proj = tf.keras.layers.Dense(self._embedding_dim)
        self._dropout = tf.keras.layers.Dropout(self._dropout_rate)

    def call(self, inputs):
        x = self._concat([h(inputs) for h in self._heads])
        x = self._proj(x)
        x = self._dropout(x)
        return x


class FeedForward(tf.keras.layers.Layer):
    """Feed forward network."""

    def __init__(self, embeddings_dim, dropout_rate=0.1, scale=4):
        super().__init__()
        self._embeddings_dim = embeddings_dim
        self._dropout_rate = dropout_rate
        self._scale = scale

    def build(self, input_shape):
        self._net = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    self._embeddings_dim * self._scale, activation="relu"
                ),
                tf.keras.layers.Dense(self._embeddings_dim),
                tf.keras.layers.Dropout(self._dropout_rate),
            ]
        )

    def call(self, inputs):
        return self._net(inputs)


class Block(tf.keras.layers.Layer):
    """Self attention block.

    It consists of multiple self-attention heads, feed forward layer and layer
    normalization."""

    def __init__(self, embeddings_dim, num_heads, dropout_rate=0.1):
        super().__init__()
        self._embeddings_dim = embeddings_dim
        self._num_heads = num_heads
        self._dropout_rate = dropout_rate

    def build(self, input_shape):
        self._sa = MultiHeadLayer(
            self._embeddings_dim, self._num_heads, dropout_rate=self._dropout_rate
        )
        self._ffwd = FeedForward(self._embeddings_dim, self._dropout_rate)
        self._ln1 = tf.keras.layers.LayerNormalization()
        self._ln2 = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self._ln1(x)
        x = x + self._sa(x)
        x = self._ln2(x)
        x = x + self._ffwd(x)
        return x


class LanguageModel(tf.keras.Model):
    """Generative language model."""

    def __init__(
        self,
        encoder,
        corpus=None,
        params=Params(),
    ):
        super().__init__()

        self._corpus = corpus
        self._token_embedding_table = tf.keras.layers.Embedding(
            encoder.vocab_size, params.embedding_dim, name="token_embedding"
        )
        self._position_embedding_table = tf.keras.layers.Embedding(
            params.block_size, params.embedding_dim, name="position_embedding"
        )
        self._blocks = tf.keras.Sequential(
            [
                Block(params.embedding_dim, params.num_heads, params.dropout_rate)
                for _ in range(params.num_blocks)
            ]
        )
        self._ln = tf.keras.layers.LayerNormalization()
        self._lm_head = tf.keras.layers.Dense(encoder.vocab_size)
        self._compute_batch_loss = BatchLoss()
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)
        self._block_size = params.block_size
        self._batch_size = params.batch_size
        self._epochs = params.epochs
        self._steps_per_epoch = params.steps_per_epoch

    def call(self, inputs):
        block_size = inputs.shape[1]
        token_embeddings = self._token_embedding_table(inputs)
        position_embeddings = self._position_embedding_table(tf.range(block_size))
        x = token_embeddings + position_embeddings
        x = self._blocks(x)
        x = self._ln(x)
        logits = self._lm_head(x)
        return logits

    def train(self, epochs=None, steps_per_epoch=None):
        if self._corpus is None:
            raise ValueError("No corpus defined for training")

        if not epochs:
            epochs = self._epochs

        if not steps_per_epoch:
            steps_per_epoch = self._steps_per_epoch

        self.compile(
            optimizer=self._optimizer, loss=self._compute_batch_loss, metrics=[]
        )
        self.fit(
            self._corpus.as_dataset(
                block_size=self._block_size, batch_size=self._batch_size
            ),
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
        )

    def generate(self, encoder, num_tokens=100, start_phrase=" "):
        """Generate sequence based on input context.

        Parameters:
        start_phrase (str): Start sequence (context).
        num_tokens (int): Number of tokens to generate.

        Returns:
        Generated text (including context) as string."""

        # Encode start sequence (str) into a 1-dimensional tensor of indices of
        # vocabulary size (tokens).
        start_v = tf.convert_to_tensor(encoder.encode(start_phrase))
        tokens = tf.reshape(start_v, (1, len(start_v)))

        # Generate next token based on existing sequence.
        for i in range(num_tokens):
            # Let the model predict the next token based on the previous sequence
            # (context). The architecture supports a context of up to 'block_size'
            # tokens.
            x = tokens[:, -self._block_size :]
            logits = self(x)

            # Extract the predicted token probabilities and use softmax as
            # normalization.
            logits = logits[:, -1, :]
            probs = tf.nn.softmax(logits, axis=-1, name=None)

            # Pick a token based on probabilities.
            # Note: Tensorflow's categorical function is taking log-probabilities.
            token_next = tf.random.categorical(tf.math.log(probs), 1, dtype=tf.int32)

            # Add the predicted token to the existing sequence.
            tokens = tf.concat([tokens, token_next], axis=1)
        return encoder.decode(tokens[0].numpy().tolist())
