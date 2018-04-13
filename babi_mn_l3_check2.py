import tarfile
import random
import sklearn.utils
import babi_data_util
import tensorflow as tf


data_path = "tasks_1-20_v1-2.tar.gz"


def create_babi_data(path, filetype="train", num_hint=1):
    tar = tarfile.open(path)
    challenge = 'tasks_1-20_v1-2/en/qa1_single-supporting-fact_{}.txt'
    return babi_data_util.get_stories(tar.extractfile(challenge.format(filetype)), num_hint)

def create_input_data_(inputs, ids, max_sentence_length, max_query_length):
    max_story_length = babi_data_util.get_max_story_length(inputs)
    ids_inputs = convert_ids_(inputs, ids, max_story_length, max_sentence_length, max_query_length)

    x = []
    q = []
    t = []
    for ids_input in ids_inputs:
        x +=[ids_input[0]]
        q += [ids_input[1]]
        t += ids_input[2]
    return x, q, t

def main():
    train_input = create_babi_data(data_path)
        test_input = create_babi_data(data_path, filetype="test", num_hint=1)

        ids, ids_ = babi_data_util.create_vocab_dict(train_input, test_input)

        max_sentence_length = max(map(len, (x for x, _, _, in train_input + test_input)))
        max_query_length = max(map(len, (x for _, x, _, in train_input + test_input)))

        x, q, t = create_input_data(train_input, ids, max_sentence_length, max_query_length)

        print(x, q, t)

if __name__ == "__main__":
    main()
