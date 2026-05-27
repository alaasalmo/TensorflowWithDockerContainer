import tensorflow as tf


def build_model(
    vectorizer,
    vocab_size
#    ,max_len
):

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(),
            dtype=tf.string
        ),

        # Text -> integer tokens
        vectorizer,

        # Embedding layer
        tf.keras.layers.Embedding(vocab_size,64),
        # GRU layers
        tf.keras.layers.GRU(64,return_sequences=True),
        tf.keras.layers.GRU(64,return_sequences=True),
        tf.keras.layers.GRU(32),
        # Dense layers
        tf.keras.layers.Dense(64,activation="relu"),
        tf.keras.layers.Dense(1,activation="sigmoid")
    ])

    return model