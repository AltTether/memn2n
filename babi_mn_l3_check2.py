import tarfile
import random
import sklearn.utils
import babi_data_util
import tensorflow as tf

data_path = "tasks_1-20_v1-2.tar.gz"


def inference(x, q, t, d, V, n_layer, batch_size):
    A = tf.Variable(tf.concat([tf.zeros(shape=[1, d], dtype=tf.float32), tf.truncated_normal(shape=[V, d], dtype=tf.float32)], axis=0))
    B = tf.Variable(tf.concat([tf.zeros(shape=[1, d], dtype=tf.float32), tf.truncated_normal(shape=[V, d], dtype=tf.float32)], axis=0))
    Cn = [tf.Variable(tf.concat([tf.zeros(shape=[1, d]), tf.truncated_normal(shape=[V, d])], axis=0)) for _ in range(n_layer)]

    next_u = tf.reduce_sum(tf.nn.embedding_lookup(B, q), axis=1) # shape=(batch_size, d)
    m = None
    for layer in range(n_layer):
        if layer == 0:
            A_ = A
        else:
            A_ = Cn[layer-1]
        C = Cn[layer]

        m = tf.reduce_sum(tf.nn.embedding_lookup(A_, x), axis=2)                # shape=(batch_size, sentence_length, d)
        p = tf.nn.softmax(tf.matmul(m, tf.reshape(next_u, shape=[-1, d, 1])))   # shape=(batch_size, sentence_length, 1)
        c = tf.reduce_sum(tf.nn.embedding_lookup(C, x), axis=2)                 # shape=(batch_size, sentence_length, d)
        o = tf.matmul(tf.transpose(p, perm=[0, 2, 1]), c)                       # shape=(batch_size, 1, d)
        next_u = tf.add(tf.reshape(o, shape=[batch_size, d]), next_u)           # shape=(batch_size, d)

    W = tf.Variable(tf.concat([tf.zeros(shape=[d, 1], dtype=tf.float32), tf.truncated_normal(shape=[d, V], dtype=tf.float32, stddev=0.1)], axis=1))
    a = tf.nn.softmax(tf.matmul(next_u, W))

    return a

def inference_(x, q, t, d, V, n_layer, batch_size):
    A = tf.Variable(tf.concat([tf.zeros(shape=[1, d]), tf.random_normal(shape=[V, d], stddev=0.1)], axis=0))
    B = tf.Variable(tf.concat([tf.zeros(shape=[1, d]), tf.random_normal(shape=[V, d], stddev=0.1], axis=0))
    Cn = [tf.Variable(tf.concat([tf.zeros(shape=[1, d]), tf.random_normal(shape=[V, d], stddev=0.1], axis=0)) for _ in range(n_layer)]

    next_u = tf.reshape(tf.reduce_sum(tf.nn.embedding_lookup(B, q), axis=1), shape=[-1, d, 1])  # shape = [bs, d, 1]

    for layer in range(n_layer):
        if layer == 0:
            A_ = A
        else:
            A_ = Cn[layer-1]
        C = Cn[layer]

        m = tf.transpose(tf.reduce_sum(tf.nn.embedding_lookup(A_, x), axis=2), perm=[0,2,1])    # shape = [bs, d, storu_l]
        p = tf.nn.softmax(tf.matmul(tf.transpose(next_u, perm=[0, 2, 1]), m))                   # shape = [bs, 1, story_l]
        c = tf.transpose(tf.reduce_sum(tf.nn.embedding_lookup(C, x), axis=2), perm=[0,2,1])     # shape = [bs, d, story_l]
        o = tf.matmul(p, tf.transpose(c, perm=[0, 2, 1]))                                       # shape = [bs, 1, d]
        next_u = tf.add(next_u, tf.transpose(o, perm=[0, 2, 1]))                                 # shape = [bs, d, 1]

    W = tf.Variable(tf.concat([tf.zeros(shape=[d, 1]), tf.random_normal(shape=[d, V], stddev=0.1)], axis=1))
    a = tf.nn.softmax(tf.matmul(tf.reshape(next_u, shape=[-1, d]), W))
    

    return a

def predict(infer_output):
    return tf.argmax(infer_output, axis=1)

def calc_loss(infer_output, t, V):
    return tf.reduce_mean(-tf.reduce_sum(tf.log(tf.clip_by_value(infer_output, 1e-10, 1.0)) * tf.one_hot(t, depth=V+1), axis=1))

def calc_loss_(infer_output, t, V):
    return -tf.reduce_sum(tf.log(tf.clip_by_value(infer_output, 1e-10, 1.0)) * tf.one_hot(t, depth=V+1))

def calc_loss__(infer_output, t, V):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(t, depth=V+1),logits=infer_output))

def train(loss):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    return optimizer.minimize(loss)

def train_(loss):
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 40)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    return optimizer.apply_gradients(zip(grads, tvars), global_step=tf.train.get_or_create_global_step())

def calc_acc(output, t):
    prediction = tf.cast(predict(output), dtype=tf.int32)
    return tf.reduce_sum(tf.cast(tf.equal(prediction, t), tf.int32)) / tf.shape(prediction)[0]

def split_train_validate_(x, q, t, train_size=0.9, is_shuffle=True):
    if is_shuffle:
        x_, q_, t_ = sklearn.utils.shuffle(x, q, t)
    n_input = len(x_)
    n_train = int(n_input * train_size)
    train_x = x_[:n_train]
    train_q = q_[:n_train]
    train_t = t_[:n_train]
    validate_x = x_[n_train:]
    validate_q = q_[n_train:]
    validate_t = t_[n_train:]
    return train_x, train_q, train_t, validate_x, validate_q, validate_t

def main():
    d = 20
    batch_size = 32
    n_layer = 1

    train_input = babi_data_util.create_babi_data(data_path)
    test_input = babi_data_util.create_babi_data(data_path, filetype="test", num_hint=1)

    ids, ids_ = babi_data_util.create_vocab_dict_(train_input, test_input)
    V = len(ids)    # 21?

    max_story_length = babi_data_util.get_max_story_length_(train_input)
    max_sentence_length = max(map(len, (x for x, _, _, in train_input + test_input)))
    max_query_length = max(map(len, (x for _, x, _, in train_input + test_input)))

    x, q, t = babi_data_util.create_input_data_(train_input, ids, max_sentence_length, max_query_length)
    x_test, q_test, t_test = babi_data_util.create_input_data_(test_input, ids, max_sentence_length, max_query_length)

    X = tf.placeholder(dtype=tf.int32, shape=[None, max_story_length, max_sentence_length])
    Q = tf.placeholder(dtype=tf.int32, shape=[None, max_query_length])
    T = tf.placeholder(dtype=tf.int32, shape=[None])
    BS = tf.placeholder(dtype=tf.int32, shape=[])

    a = inference_(X, Q, T, d, V, n_layer, BS)
    loss = calc_loss__(a, T, V)
    train_step = train_(loss)

    accuracy = calc_acc(a, T)
    prediction = predict(a)

    n_batch = len(x) // batch_size

    sess = tf.Session()
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state('./checkpoints/')
    if ckpt:
        last_model = ckpt.model_checkpoint_path
        saver.restore(sess, last_model)
    else:
        sess.run(tf.global_variables_initializer())

    for epoch in range(10000000):
        x_train, q_train, t_train, x_valid, q_valid, t_valid = babi_data_util.split_train_validate_(x, q, t)
        n_batch = len(x_train) // batch_size
        for i in range(n_batch):
            start = batch_size * i
            end = start + batch_size
            sess.run(train_step, feed_dict={X: x_train[start:end], Q: q_train[start:end], T: t_train[start:end], BS: batch_size})
        if epoch % 10 == 0:
            print("train_loss: {0:.10f}".format(loss.eval(session=sess, feed_dict={X: x_train, Q: q_train, T: t_train, BS: len(x_train)})),
                  "valid_loss: {0:.3f}".format(loss.eval(session=sess, feed_dict={X: x_valid, Q: q_valid, T: t_valid, BS: len(x_valid)})),
                  "train_acc: {0:.4f}".format(accuracy.eval(session=sess, feed_dict={X: x_train, Q: q_train, T: t_train, BS: len(x_train)})),
                  "valid_acc: {0:.4f}".format(accuracy.eval(session=sess, feed_dict={X: x_valid, Q: q_valid, T: t_valid, BS: len(x_valid)})))
            print(prediction.eval(session=sess, feed_dict={X: x_train[0:10], Q: q_train[0:10], T: t_train[0:10], BS: len(x_train[0:10])}))
            print(t_train[0:10])

        if epoch % 100 == 0 and epoch != 0:
            saver.save(sess, "./checkpoints/model.ckpt")


if __name__ == "__main__":
    main()
