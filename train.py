import sys as sys
import tensorflow as tf

import corpus
import model

# Parse arguments.
argv = sys.argv[1:]
filename = argv[0]
out_filename = argv[1]

# Read training data.
text = ""
with open(filename, "r", encoding="utf8") as f:
    text = f.read()

encoder = corpus.CharacterEncoder(text)
corpus = corpus.TextCorpus(text, encoder=encoder)

# Train the model and save the weights.
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
m = model.LanguageModel(encoder, corpus=corpus)
m.train()
m.save_weights(out_filename, save_format="tf")
