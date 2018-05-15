import tarfile
import sklearn.utils


def get_stories(inputs, num_hint):
    inputs_list = to_list(inputs)
    sp_inputs_list = filter_and_split(inputs_list)
    return to_input_data__(sp_inputs_list, num_hint)

def to_list(inputs):
    return [data.decode('utf-8') for data in inputs]

def filter_and_split(inputs):
    return [data.rstrip().replace("\t", "", 1).replace("\t", " ", 1).replace(".", " .", 1).replace("?", " ?", 1).split(" ") for data in inputs]

def to_input_data(inputs):
    return_value = []
    qa_dataset = []
    qa_dataset_tmp = []
    for i, data in enumerate(inputs):
        sentence_i = i % 15 + 1
        if sentence_i % 3 != 0:
            qa_dataset += data
        else:
            qa_dataset_tmp += qa_dataset
            return_value += [(qa_dataset_tmp, data[:len(data)-3], data[len(data)-2:len(data)-1])]
            qa_dataset_tmp = list()
        if sentence_i == 15:
            qa_dataset = list()
    return return_value

def to_input_data_(inputs, num_hint):
    return_value = list()
    qa_dataset = list()
    qa_dataset_tmp = list()
    previous_sentence_number = 0
    current_sentence_number = 0
    for data in inputs:
        current_sentence_number = int(data.pop(0))
        if previous_sentence_number >= current_sentence_number:
            qa_dataset = list()
            qa_dataset_tmp = list()
        if "?" not in data:
            qa_dataset += data
        else:
            qa_dataset_tmp += qa_dataset
            return_value += [(qa_dataset_tmp, data[:len(data)-num_hint-1], data[len(data)-num_hint-1:len(data)-num_hint])]
            qa_dataset_tmp = list()
        previous_sentence_number = current_sentence_number
    return return_value

def to_input_data__(inputs, num_hint):
    return_value = list()
    qa_dataset = list()
    qa_dataset_tmp = list()
    previous_sentence_number = 0
    current_sentence_number = 0
    for data in inputs:
        current_sentence_number = int(data.pop(0))
        if previous_sentence_number >= current_sentence_number:
            qa_dataset = list()
            qa_dataset_tmp = list()
        if "?" not in data:
            qa_dataset += [data]
        else:
            qa_dataset_tmp += qa_dataset
            return_value += [(qa_dataset_tmp, data[:len(data)-num_hint-1], data[len(data)-num_hint-1:len(data)-num_hint])]
            qa_dataset_tmp = list()
        previous_sentence_number = current_sentence_number
    return return_value

def get_max_story_length(inputs):
    max_story_length = -1
    story_length_count = 0
    previous_sentence_number = 0
    current_sentence_number = 0
    for data in inputs:
        current_sentence_number = int(data.pop(0))
        if previous_sentence_number >= current_sentence_number:
            max_story_length = max(max_story_length, story_length_count)
            story_length_count = 0
        previous_sentence_number = current_sentence_number
        story_length_count += 1
    return max_story_length

def get_max_story_length_(inputs):
    max_story_length = -1;
    for story in inputs:
        max_story_length = max(max_story_length, len(story[0]))
    return max_story_length


def create_vocab_dict(train_input, test_input):
    vocab = set()
    for story, q, answer in train_input + test_input:
        vocab |= set(story + q + answer)
    vocab = sorted(vocab)
    ids = dict((v, i+1) for i, v in enumerate(vocab))
    ids_ = [0 for _ in range(len(vocab)+1)]
    for i, v in enumerate(vocab):
        ids_[i+1] = v
    return ids, ids_

def create_vocab_dict_(train_input, test_input):
    vocab = set()
    for story, q, answer in train_input + test_input:
        story_ = []
        for sentence in story:
            story_ += sentence
        vocab |= set(story_ + q + answer)
    vocab = sorted(vocab)
    ids = dict((v, i+1) for i, v in enumerate(vocab))
    ids_ = [0 for _ in range(len(vocab)+1)]
    for i, v in enumerate(vocab):
        ids_[i+1] = v
    return ids, ids_

def create_babi_data(path, filetype="train", num_hint=1):
    tar = tarfile.open(path)
    challenge = 'tasks_1-20_v1-2/en/qa1_single-supporting-fact_{}.txt'
    return get_stories(tar.extractfile(challenge.format(filetype)), num_hint)

def create_input_data_(inputs, ids, max_sentence_length, max_query_length):
    max_story_length = get_max_story_length_(inputs)
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
