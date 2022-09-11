import numpy as np
import numpy.random
import argparse, re
from random import randint


def get_word_index(word, vocab):
    return np.where(vocab == word)[0][0]


def generate_sequence(length, inp, vocab):
    if inp.size == length:
        return inp

    result = []
    for word in vocab:
        temp = generate_sequence(length, np.append(inp, word), vocab)
        result.append(temp)
    return result


def MLE(inp, word, vocab, data_table):
    indicies = tuple([get_word_index(i, vocab) for i in inp])

    return data_table[indicies] / numpy.sum(data_table[get_word_index(word, vocab)])


def predict(length, prefix="", n=2, model="model.npy"):
    data_table = np.load(model)

    words = np.load("vocabulary.npy")
    if prefix == "":
        prefix = [words[randint(0, words.size)]]
    else:
        prefix = prefix.split()
    i = 0
    while len(prefix) < n - 1:
        prefix.append(words[randint(0, words.size)])

    probs = []
    for word in words:

        probs.append(MLE(prefix + [word], word, words, data_table))
    mx = max(probs)
    mx_ind = probs.index(mx)
    prefix.append(words[mx_ind])
    i = n - 1
    while len(prefix) < length:
        probs = []
        for word in words:
            probs.append(MLE([prefix[i], word], word, words, data_table))
        mx = max(probs)
        mx_ind = probs.index(mx)
        prefix.append(words[mx_ind])
        i += 1
    print(prefix)

def main():
    parser = argparse.ArgumentParser(description='type dir to in.txt and n parameter ')
    parser.add_argument('--prefix', type=str, help='Your string', required=False, default="")
    parser.add_argument('--n', type=int, help='N for n-gram model', default=2, required=False)
    parser.add_argument('--model', type=str, help='path to load pre-trained model', default="model.npy", required=False)
    parser.add_argument('--length', type=int, help='length of sequence', required=False)
    args = parser.parse_args()
    predict(prefix=args.prefix, n=args.n, model=args.model, length=args.length)


if __name__ == "__main__":
    main()
