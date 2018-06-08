import tarfile
import random
import sklearn.utils
import babi_data_util
import tensorflow as tf

data_path = "tasks_1-20_v1-2.tar.gz"

def inference(x, q, t, V, d, n_layer, max_story_l):
    A = tf.Variable(tf.concat([tf.zeros(shape=[1, d]), tf.random_normal(shape=[V ,d], stddev=0.1)], axis=0))
    B = tf.Variable(tf.concat([tf.zeros(shape=[1, d]), tf.random_normal(shape=[V ,d], stddev=0.1)], axis=0))
    Cn = [tf.Variable(tf.concat([tf.zeros(shape=[1, d]), tf.random_normal(shape=[V ,d], stddev=0.1)], axis=0)) for _ in range(n_layer)]
    W = tf.Variable(tf.concat([tf.zeros(shape=[d, 1]), tf.random_normal(shape=[d, V], stddev=0.1)], axis=1))
    Ta = tf.Variable(tf.random_normal(shape=[max_story_l, d], stddev=0.1))
    Tcn = [tf.Variable(tf.random_normal(shape=[max_story_l, d], stddev=0.1)) for _ in range(n_layer)]

    next_u = tf.reduce_sum(tf.nn.embedding_lookup(B, q), axis=1)      # shape=[bs,d]

    for i in range(n_layer):
        if i == 0:
            A_ = A
            Ta_ = Ta
        else:
            A_ = Cn[i-1]
            Ta_ = Tcn[i-1]
        C = Cn[i]
        Tc = Tcn[i]

        m = tf.reduce_sum(tf.nn.embedding_lookup(A_, x), axis=2)
        m_Ta = (m + Ta) * tf.abs(tf.sign(m))
        p = tf.nn.softmax(tf.matmul(m_Ta, tf.reshape(next_u, shape=[-1,d,1])))
        c = tf.reduce_sum(tf.nn.embedding_lookup(C, x), axis=2)
        c_Tc = (c + Tc) * tf.abs(tf.sign(c))
        o = tf.matmul(tf.transpose(c_Tc, perm=[0,2,1]), p)
        next_u = tf.add(tf.reshape(o, shape=[-1,d]), next_u)

    return tf.matmul(next_u, W)


def calc_loss(infer_output, t, V):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(t, depth=V+1), logits=infer_output))

def train(loss, learning_rate):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_step = optimizer.minimize(loss)
    return train_step

def predict(infer_output):
    return tf.cast(tf.argmax(infer_output, axis=1), dtype=tf.int32)

def calc_acc(output, t):
    prediction = tf.cast(predict(output), tf.int32)
    return tf.reduce_sum(tf.cast(tf.equal(prediction, t), tf.int32)) / tf.shape(prediction)[0]

def main():
    d = 20
    batch_size = 32
    n_layer = 1
    learning_rate = 0.01

    train_input = babi_data_util.create_babi_data(data_path)
    test_input = babi_data_util.create_babi_data(data_path, filetype="test", num_hint=1)

    ids, ids_ = babi_data_util.create_vocab_dict_(train_input, test_input)
    V = len(ids)

    max_story_length = babi_data_util.get_max_story_length_(train_input)
    max_sentence_length = max(map(len, (x for x, _, _, in train_input + test_input)))
    max_query_length = max(map(len, (x for _, x, _, in train_input + test_input)))

    x, q, t = babi_data_util.create_input_data_(train_input, ids, max_sentence_length, max_query_length)
    x_test, q_test, t_test = babi_data_util.create_input_data_(test_input, ids, max_sentence_length, max_query_length)

    X = tf.placeholder(dtype=tf.int32, shape=[None, max_story_length, max_sentence_length])
    Q = tf.placeholder(dtype=tf.int32, shape=[None, max_query_length])
    T = tf.placeholder(dtype=tf.int32, shape=[None])

    a = inference(X, Q, T, V, d, n_layer, max_story_length)

    loss = calc_loss(a, T, V)
    train_step = train(loss, learning_rate)
    prediction = predict(a)
    acc = calc_acc(a, T)

    sess = tf.Session()
    #saver = tf.train.Saver()
    #ckpt = tf.train.get_checkpoint_state('./checkpoints/')
    #if ckpt:
    #    last_model = ckpt.model_checkpoint_path
    #    saver.restore(sess, last_model)
    #else:
    sess.run(tf.global_variables_initializer())

    for epoch in range(10000):
        x_train, q_train, t_train, x_valid, q_valid, t_valid = babi_data_util.split_train_validate_(x[0:100], q[0:100], t[0:100])
        n_batch = len(x_train) // batch_size
        for i in range(n_batch):
            start = batch_size * i
            end = start + batch_size
            sess.run(train_step, feed_dict={X: x_train[start:end], Q: q_train[start:end], T: t_train[start:end]})

        train_loss = loss.eval(session=sess, feed_dict={X: x_train, Q: q_train, T: t_train})
        valid_loss = loss.eval(session=sess, feed_dict={X: x_valid, Q: q_valid, T: t_valid})
        train_acc = acc.eval(session=sess, feed_dict={X: x_train, Q: q_train, T: t_train})
        valid_acc = acc.eval(session=sess, feed_dict={X: x_valid, Q: q_valid, T: t_valid})
        valid_pred = prediction.eval(session=sess, feed_dict={X: x_valid, Q: q_valid, T: t_valid})

        print(train_loss, valid_loss, train_acc, valid_acc)
        for y_, y in zip([ids_[id_] for id_ in t_valid], [ids_[id_] for id_ in valid_pred]):
            print(y_, y)


if __name__ == "__main__":
    main()
