TF_MODEL_NAME=model.tf
INPUT_FILENAME=input.txt

# Note: Make sure to set up your conda environment accordingly.

$(INPUT_FILENAME):
	wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

$(TF_MODEL_NAME): train.py $(INPUT_FILENAME)
	python3 train.py $(INPUT_FILENAME) $(TF_MODEL_NAME)

train: $(TF_MODEL_NAME)

gen: generate.py $(TF_MODEL_NAME).index $(INPUT_FILENAME)
	python3 generate.py $(INPUT_FILENAME) $(TF_MODEL_NAME) 200 " "

setup:
	conda install -c apple tensorflow-deps
	pip install tensorflow-macos
	pip install tensorflow-metal
	pip install git+https://github.com/psf/black
	pip install mdformat

format:
	black *.py
	mdformat *.md

clean:
	rm -r $(TF_MODEL_NAME)* $(INPUT_FILENAME)

edit:
	vim *.py Makefile *.md checkpoint

all: train

