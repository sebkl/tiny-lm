import tensorflow as tf
import sys as sys

import model
import corpus

# Parse arguments.
argv = sys.argv[1:]
filename = argv[0]
model_filename = argv[1]
num_tokens = int(argv[2])
start_phrase = argv[3]

# Read training data.
text = ""
with open(filename, "r", encoding="utf8") as f:
    text = f.read()

encoder = corpus.CharacterEncoder(text)

m = model.LanguageModel(encoder)

m.load_weights(model_filename)

print("Generating %d tokens for start phrase '%s':\n" % (num_tokens, start_phrase))
print(m.generate(encoder, num_tokens, start_phrase=start_phrase))
