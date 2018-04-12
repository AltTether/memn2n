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
