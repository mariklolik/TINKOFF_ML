import numpy as np
import numpy.random
import argparse, re


def generate_ngrams(text, n=2):
    return np.array([text[i: i + n] for i in range(0, text.size - n)])

def get_word_index(word, vocab):
    return np.where(vocab == word)[0][0]

def train(input_data, output, n=2):
    text = ""
    with open(input_data, "r", encoding="UTF-8") as file:
        for i in file:
            text += i
    text = text.lower()
    text = re.split('[^a-zа-яё]+', text, flags=re.IGNORECASE)
    text = np.array(text, dtype=str)
    text = text[text != '']

    words = np.unique(text)
    words.sort()
    words_size = words.size
    # Table to save frequencies of n-grams (shaped words_size * words_size * ... * words_size n - times)
    data_table = np.array([0] * words_size ** n, dtype=np.int64).reshape(
        *[
             words_size] * n)

    ngrams = generate_ngrams(text, n)
    for ngr in ngrams:
        indices = [get_word_index(word, words) for word in ngr]
        data_table[tuple(indices)] += 1
    np.save("vocabulary.npy", words)
    np.save(output, data_table)


def main():
    parser = argparse.ArgumentParser(description='type dir to in.txt and n parameter ')
    parser.add_argument('--input_dir', type=str, help='Input dir for in.txt', required=True)
    parser.add_argument('--n', type=int, help='N for n-gram model', default=2, required=False)
    parser.add_argument('--model', type=str, help='path to save model', default="model.npy", required=False)
    args = parser.parse_args()
    if args.input_dir == "":
        raise Exception
    train(input_data=args.input_dir, output=args.model, n=args.n, )


if __name__ == "__main__":
    main()
