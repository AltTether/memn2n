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
    max_story_length = babi_data_util.get_max_story_length_(inputs)
    ids_inputs = convert_ids_(inputs, ids, max_story_length, max_sentence_length, max_query_length)

    x = []
    q = []
    t = []
    for ids_input in ids_inputs:
        x +=[ids_input[0]]
        q += [ids_input[1]]
        t += ids_input[2]
    return x, q, t

def convert_ids_(inputs, ids, max_story_length, max_sentence_length, max_query_length):
    output = []
    for x, q, t in inputs:
        x_ids_list = []
        for sentence in x:
            sentence_ids = []
            for word in sentence:
                sentence_ids += [ids[word]]
            if len(sentence_ids) < max_sentence_length:
                sentence_ids += (0 for _ in range(max_sentence_length - len(sentence_ids)))

            x_ids_list += [sentence_ids]

        if len(x_ids_list) < max_story_length:
            for _ in range(max_story_length - len(x_ids_list)):
                x_ids_list += [[0 for __ in range(max_sentence_length)]]

        q_ids = []
        for word in q:
            q_ids += [ids[word]]
        if len(q_ids) < max_query_length:
            q_ids += (0 for _ in range(max_query_length - len(q_ids)))
        t_id = [ids[t[0]]]
        output += [[x_ids_list, q_ids, t_id]]
    return output

def inference(x, q, t, d, V, n_layer):
    A = tf.Variable(tf.concat([tf.zeros(shape=[1, d], dtype=tf.float32), tf.truncated_normal(shape=[V, d], dtype=tf.float32)], axis=0))
    Cn = [tf.Variable(tf.concat([tf.zeros(shape=[1, d]), tf.truncated_normal(shape=[V, d])], axis=0)) for _ in range(n_layer)]

    next_u = tf.nn.embedding_lookup(A, q)
    return next_u

def main():
    V = 20
    d = 20
    batch_size = 32
    n_layer = 3

    train_input = create_babi_data(data_path)
    test_input = create_babi_data(data_path, filetype="test", num_hint=1)

    ids, ids_ = babi_data_util.create_vocab_dict_(train_input, test_input)

    max_story_length = babi_data_util.get_max_story_length_(train_input)
    max_sentence_length = max(map(len, (x for x, _, _, in train_input + test_input)))
    max_query_length = max(map(len, (x for _, x, _, in train_input + test_input)))

    x, q, t = create_input_data_(train_input, ids, max_sentence_length, max_query_length)

    X = tf.placeholder(dtype=tf.int32, shape=[None, max_story_length, max_sentence_length])
    Q = tf.placeholder(dtype=tf.int32, shape=[None, max_query_length])
    T = tf.placeholder(dtype=tf.int32, shape=[None])

    next_u = inference(X, Q, T, d, V, n_layer)

    n_batch = len(x) // batch_size

    sess = tf.Session();
    sess.run(tf.global_variables_initializer())

    for epoch in range(1):
        for i in range(n_batch):
            start = batch_size * i
            end = start + batch_size
            sess.run(tf.shape(next_u), feed_dict={X: x[start:end], Q: q[start:end], T: t[start:end]})
if __name__ == "__main__":
    main()
