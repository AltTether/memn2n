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

def create_input_data(inputs, ids, max_sentence_length, max_query_length):
    ids_inputs = convert_ids(inputs, ids, max_sentence_length, max_query_length)

    x = []
    q = []
    t = []
    for ids_input in ids_inputs:
        x +=[ids_input[0]]
        q += [ids_input[1]]
        t += ids_input[2]
    return x, q, t

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

def convert_ids(inputs, ids, max_sentence_length, max_query_length):
    one_data = []
    for x, q, t in inputs:
        x_ids_list = []
        sentence_ids = []
        for word in x:
            sentence_ids += [ids[word]]
        if len(sentence_ids) < max_sentence_length:
            sentence_ids += (0 for _ in range(max_sentence_length - len(sentence_ids)))
        x_ids_list += [sentence_ids]
        q_ids = []
        for word in q:
            q_ids += [ids[word]]
        if len(q_ids) < max_query_length:
            q_ids += (0 for _ in range(max_query_length - len(q_ids)))
        t_id = [ids[t[0]]]
        one_data += [[x_ids_list, q_ids, t_id]]
    return one_data

def convert_ids_(inputs, ids, max_story_length, max_sentence_length, max_query_length):
    one_data = []
    for x, q, t in inputs:
        x_ids_list = []
        sentence_ids = []
        for sentence in x:
            for word in sentence:
                sentence_ids += [ids[word]]
            if len(sentence_ids) < max_sentence_length:
                sentence_ids += (0 for _ in range(max_sentence_length - len(sentence_ids)))
            
        x_ids_list += [sentence_ids]

        q_ids = []
        for word in q:
            q_ids += [ids[word]]
        if len(q_ids) < max_query_length:
            q_ids += (0 for _ in range(max_query_length - len(q_ids)))
        t_id = [ids[t[0]]]
        one_data += [[x_ids_list, q_ids, t_id]]
    return one_data

def split_train_validate(x, q, t, train_size=0.9):
    n_input = len(x)
    n_train = int(n_input * train_size)
    train_x = x[:n_train]
    train_q = q[:n_train]
    train_t = t[:n_train]
    validate_x = x[n_train:]
    validate_q = q[n_train:]
    validate_t = t[n_train:]
    return train_x, train_q, train_t, validate_x, validate_q, validate_t

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

"""
1層MemN2Nモデル
学習はacc50%まで可能であった
"""
def l1_inference(x, q, t, batch_size, V, d):
    A = tf.Variable(tf.concat([tf.zeros(shape=[1, d]), tf.random_normal(shape=[V, d])], axis=0))
    B = tf.Variable(tf.concat([tf.zeros(shape=[1, d]), tf.random_normal(shape=[V, d])], axis=0))
    C = tf.Variable(tf.concat([tf.zeros(shape=[1, d]), tf.random_normal(shape=[V, d])], axis=0))

    m = tf.reshape(tf.reduce_sum(tf.nn.embedding_lookup(A, x), axis=1), shape=[batch_size, d, 1]) # shape=[batch_size, d]
    u = tf.reshape(tf.reduce_sum(tf.nn.embedding_lookup(B, q), axis=1), shape=[batch_size, d, 1]) # shape=[batch_size, d]
    c = tf.reshape(tf.reduce_sum(tf.nn.embedding_lookup(C, x), axis=1), shape=[batch_size, d, 1]) # shape=[batch_size, d]

    p = tf.nn.softmax(tf.matmul(tf.transpose(u, perm=[0,2,1]), m)) # shape=[batch_size, 1, 1]
    o = tf.reshape(tf.matmul(c, p), shape=[batch_size, d]) # shape=[batch_size, d, 1]

    W = tf.Variable(tf.random_normal(shape=[d, V+1]))
    a = tf.nn.softmax(tf.matmul(tf.add(o, tf.reshape(u, shape=[batch_size, d])), W))
    return a

"""
3層MemN2Nモデル
lossが下がってerrorが上がるのでおそらくバグあり
"""
def l3_inference(X, Q, T, d, V, batch_size, n_query, n_layer, learning_rate):
    A = tf.Variable(tf.concat([tf.zeros(shape=[d, 1]), tf.random_normal([d, V], stddev=0.1)], axis=1))
    Cn = [tf.Variable(tf.concat([tf.zeros(shape=[d, 1]), tf.random_normal([d, V], stddev=0.1)], axis=1)) for _ in range(n_layer)]
    W = tf.Variable(tf.random_normal([V+1, d], stddev=0.1))

    embed_X_B = tf.nn.embedding_lookup(tf.transpose(A), Q)
    next_u = tf.reduce_sum(embed_X_B, 1)
    for layer_nth in range(n_layer):
        if layer_nth == 0:
            A_ = A
        else:
            A_ = Cn[layer_nth-1]
        C = Cn[layer_nth]
        next_u = calc_o_u(A_, C, X, next_u, batch_size, n_query, d)
    a = tf.transpose(tf.nn.softmax((tf.matmul(W, tf.transpose(next_u)))))
    loss = calc_loss(a, T, V)
    train_step = train(loss, learning_rate)
    return loss, train_step, a

"""
３層MemN2Nモデルのテストバージョン
"""
def inference(X, Q, T, d, V, batch_size, n_layer):
    A = tf.Variable(tf.concat([tf.zeros(shape=[1, d]), tf.random_normal(shape=[V, d])], axis=0))
    Cn = [tf.Variable(tf.concat([tf.zeros(shape=[1, d]), tf.random_normal(shape=[V, d])], axis=0)) for _ in range(n_layer)]

    next_u = tf.reshape(tf.reduce_sum(tf.nn.embedding_lookup(A, Q), axis=1), shape=[batch_size, d, 1]) # shape=[batch_size, d]

    for layer_nth in range(n_layer):
        if layer_nth == 0:
            A_ = A
        else:
            A_ = Cn[layer_nth-1]
        C = Cn[layer_nth]
        next_u = tf.reshape(calc_o_u_(X, A_, C, next_u, batch_size, d), shape=[batch_size, d, 1])

    a = tf.nn.softmax(tf.matmul(tf.reshape(next_u, shape=[batch_size, d]), tf.transpose(Cn[n_layer-1])))
    return a

def calc_o_u_(x, A, C, u, batch_size, d):
    m = tf.reshape(tf.reduce_sum(tf.nn.embedding_lookup(A, x), axis=1), shape=[batch_size, d, 1]) # shape=[batch_size, d]
    c = tf.reshape(tf.reduce_sum(tf.nn.embedding_lookup(C, x), axis=1), shape=[batch_size, d, 1]) # shape=[batch_size, d]

    p = tf.nn.softmax(tf.matmul(tf.transpose(u, perm=[0,2,1]), m)) # shape=[batch_size, 1, 1]
    o = tf.reshape(tf.matmul(c, p), shape=[batch_size, d]) # shape=[batch_size, d, 1]
    return tf.add(o, tf.reshape(u, shape=[batch_size, d]))

def calc_o_u(A, C, X, u, batch_size, n_query, d):
    embed_X_A = tf.nn.embedding_lookup(tf.transpose(A), X)
    m = tf.reduce_sum(embed_X_A, 2)
    embed_X_C = tf.nn.embedding_lookup(tf.transpose(C), X)
    c = tf.reduce_sum(embed_X_C, 2)
    batch_u = tf.reshape(u, shape=[batch_size, n_query, d])
    transpose_batch_u = tf.transpose(batch_u, perm=[0, 2, 1])
    p = tf.nn.softmax(tf.matmul(m, transpose_batch_u))
    o = tf.reshape(tf.matmul(tf.transpose(p, perm=[0, 2, 1]), c), shape=[batch_size, d])
    o_u = tf.add(o, u)
    return o_u

def calc_loss(output, t, V):
    return tf.reduce_mean(-tf.reduce_sum(tf.log(tf.clip_by_value(output, 1e-10, 1.0)) * tf.one_hot(t, depth=V+1), axis=1))

def train(loss, learning_rate):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    #optimizer = tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.9, beta2=0.999)
    train_step = optimizer.minimize(loss)
    return train_step

"""
normを40におさえる用に変更したtrain
"""
def train_(loss, learning_rate):
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 40)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    #optimizer = tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.9, beta2=0.999)
    return optimizer.apply_gradients(zip(grads, tvars),global_step=tf.train.get_or_create_global_step())


def predict(output):
    return tf.argmax(output, axis=1)

def calc_error(output, t):
    prediction = tf.cast(predict(output), tf.int32)
    return tf.reduce_sum(tf.sign(tf.abs(tf.subtract(prediction, t)))) / tf.shape(t)[0]

def calc_error_(output, t):
    return 1-calc_acc(output, t)

def calc_acc(output, t):
    prediction = tf.cast(predict(output), tf.int32)
    return tf.reduce_sum(tf.cast(tf.equal(prediction, t), tf.int32)) / tf.shape(t)[0]

def main():
    input_size = 1
    V = 21
    d = 20
    n_sentence = 1
    n_query = 1
    n_layer = 3
    batch_size = 32
    train_size = 0.9
    learning_rate = 0.01

    train_input = create_babi_data(data_path, filetype="train", num_hint=1)
    test_input = create_babi_data(data_path, filetype="test", num_hint=1)

    ids, ids_ = babi_data_util.create_vocab_dict(train_input, test_input)

    max_sentence_length = max(map(len, (x for x, _, _, in train_input + test_input)))
    max_query_length = max(map(len, (x for _, x, _, in train_input + test_input)))

    X = tf.placeholder(dtype=tf.int32, shape=[None, 1, max_sentence_length])
    X_ = tf.reshape(X, shape=[-1, max_sentence_length])
    Q = tf.placeholder(dtype=tf.int32, shape=[None, max_query_length])
    T = tf.placeholder(dtype=tf.int32, shape=[None])
    LR = tf.placeholder(dtype=tf.int32, shape=())
    BS = tf.placeholder(dtype=tf.int32, shape=())

    #loss, train_step, output = inference(X, Q, T, d, V, BS, n_query, n_layer, LR)
    a = inference(X_, Q, T, d, V, BS, n_layer)
    loss = calc_loss(a, T, V)
    train_step = train_(loss, LR)
    predictions = predict(a)
    error = calc_error_(a, T)

    sess = tf.Session()

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state('./checkpoint/')
    if ckpt:
        last_model = ckpt.model_checkpoint_path
        saver.restore(sess, last_model)
    else:
        sess.run(tf.global_variables_initializer())

    x, q, t = create_input_data(train_input, ids, max_sentence_length, max_query_length)
    for epoch in range(20000000000+1):
        train_x, train_q, train_t, validate_x, validate_q, validate_t = split_train_validate_(x, q, t, train_size=train_size, is_shuffle=True)
        n_batch = int((len(x)*train_size) / batch_size)
        if epoch % 25 == 0 and epoch != 0:
            learning_rate /= 2
        for i in range(n_batch):
            start = batch_size * i
            end = start + batch_size
            """
            for sentence in train_x[start:end]:
                for sentence_ in sentence:
                    print([ids_[word] for word in sentence_])
            for query in train_q[start:end]:
                print([ids_[word] for word in query])
            print([ids_[word] for word in train_t[start:end]])
            """
            sess.run(train_step, feed_dict={X:train_x[start:end], Q:train_q[start:end], T:train_t[start:end], LR: learning_rate, BS: batch_size})
        if epoch % 1 == 0:
            print("{}epoch".format(epoch))
            print("train_loss: {0:.10f}  valid_loss: {1:.10f}".format(loss.eval(session=sess, feed_dict={X:train_x, Q:train_q, T:train_t, LR: learning_rate, BS: len(train_x)}),
                                                          loss.eval(session=sess, feed_dict={X:validate_x, Q:validate_q, T:validate_t, LR: learning_rate, BS: len(validate_x)})))
            print("train_error: {0:.3f}  valid_error: {1:.3f}".format(error.eval(session=sess, feed_dict={X:train_x, Q:train_q, T:train_t, LR: learning_rate, BS: len(train_x)}),
                                                            error.eval(session=sess, feed_dict={X:validate_x, Q:validate_q, T:validate_t, LR: learning_rate, BS: len(validate_x)})))
            #for pred, ans in zip(predictions.eval(session=sess, feed_dict={X:validate_x[0:10], Q:validate_q[0:10], T:validate_t[0:10], LR: learning_rate, BS: 10}), validate_t[0:10]):
            #    print("predict: {}, ans: {}".format(ids_[pred], ids_[ans]))

        if epoch % 100 == 0 and epoch != 0:
            saver.save(sess, "./checkpoint/d_20_norm_model.ckpt")


if __name__ == "__main__":
    main()
