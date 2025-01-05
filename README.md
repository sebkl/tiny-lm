# A simple Tensorflow based generative language model

A simple generative language model trained using a Tensorflow based deep neural network.
It is an adaption of [2] using Tensorflow instead of PyTorch for the sake of an exercise.

## Motivation

Purpose of this example is solely for me to understand the basics of training a transformer.
If you find this helpful, great. If you already know most of it, even better.

For sure there are plenty of things done wrong here or at least could be improved to some extent.
Please feel free to let me know.

## How To

The included `Makefile` serves as an example how to set up, train and run the model.
Please adjust this to your needs. It uses Anaconda as a package and environment manager.

```sh
make train
make gen
```

## References

1. [Andrej Karpathy's "Let's build GPT: from scratch, in code, spelled out."](https://www.youtube.com/watch?v=kCc8FmEb1nY)
1. [GPT video lecture](https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py)
1. [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
1. [Dropout regularization](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)
1. [Deep residual networks](https://arxiv.org/pdf/1512.03385)
1. [Layer Normalization](https://arxiv.org/pdf/1607.06450)
1. [Tensorflow](https://www.tensorflow.org)
1. [PyTorch](https://pytorch.org/)
1. [Anaconda](https://anaconda.org/anaconda/conda)

## TODOs

- Fix usage of validation data set.
- Add scripts or any other reasonable support to run training on Google Cloud Platform.
- Add more documentation as to why this neural network architecture is chosen.
- Experiment with sentence piece as a tokenizer.
- Make neural network architecture more Tensorflow idiomatic.
