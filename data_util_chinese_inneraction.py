# -*-encoding:utf-8-*-

import numpy as np
import os
import codecs
import pynlpir
from pynlpir import nlpir
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


class WordSegment(object):
    def __init__(self):
        '''初始化nlpir资源'''
        if not nlpir.Init(nlpir.PACKAGE_DIR, pynlpir.nlpir.UTF8_CODE, None):
            print('Initialize NLPIR failed')
            exit(-11111)

    def fileProcess(self, sFile, rFile, isPOS):
        '''
        对文档分词：sFile,待分词文档路径
                   rFile,分词结果文档路径
                   isPOS,是否标注词性p（1 是，0 否）
        '''
        nlpir.SetPOSmap(nlpir.ICT_POS_MAP_SECOND)
        if nlpir.FileProcess(sFile, rFile, isPOS) == 0:
            print
            'FileProcess failed.Traceback to module:word_segmentation.py,function:fileProcess()'
            exit(-111111)

    def  ParagraphProcess(self, strs):
        nlpir.ParagraphProcess(strs, False)

    def import_userdict(self, dict_path):
        nlpir.ImportUserDict(dict_path)

    def import_AddUserWord(self, word):
        nlpir.AddUserWord(str(word))

    def finalizeR(self):
        '''释放nlpir资源'''
        nlpir.Exit()


def split_data_train_test(sentence, targets, labels, locations, test_size, shuffer=True):
    np.random.seed(42)
    sentence = np.asarray(sentence)
    targets = np.asarray(targets)
    labels = np.asarray(labels)
    locations = np.asarray(locations)
    data_size = len(sentence)
    if shuffer is True:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_sentence = sentence[shuffle_indices]
        shuffled_targets = targets[shuffle_indices]
        shuffled_labels = labels[shuffle_indices]
        shuffled_locations = locations[shuffle_indices]
    else:
        shuffled_sentence = sentence
        shuffled_targets = targets
        shuffled_labels = labels
        shuffled_locations = locations

    index = int(data_size*test_size)
    train_shuffled_sentence = shuffled_sentence[:-index]
    test_shuffled_sentence = shuffled_sentence[-index:]
    train_shuffled_targets = shuffled_targets[:-index]
    test_shuffled_targets = shuffled_targets[-index:]
    train_shuffled_labels = shuffled_labels[:-index]
    test_shuffled_labels = shuffled_labels[-index:]
    train_shuffled_locations = shuffled_locations[:-index]
    test_shuffled_locations = shuffled_locations[-index:]

    return train_shuffled_sentence, train_shuffled_targets, train_shuffled_labels, train_shuffled_locations, \
           test_shuffled_sentence, test_shuffled_targets, test_shuffled_labels, test_shuffled_locations


def read_datasets(domain_name, data_base_dir):
    sentences_list = []
    targets_list = []
    labels_list = []
    locations_list = []
    sentence_file_name = str(domain_name) + "_sentence.txt"
    target_file_name = str(domain_name) + "_target.txt"
    label_file_name = str(domain_name) + "_label.txt"
    sentence_dir = os.path.join(data_base_dir, sentence_file_name)
    target_dir = os.path.join(data_base_dir, target_file_name)
    label_dir = os.path.join(data_base_dir, label_file_name)
    sentences = codecs.open(sentence_dir, 'r', encoding='utf-8').readlines()
    targets = codecs.open(target_dir, 'r', encoding='utf-8').readlines()
    labels = codecs.open(label_dir, 'r', encoding='utf-8').readlines()
    wg = WordSegment()
    for target in targets:
        for a in target.strip().split(" "):
            wg.import_AddUserWord(a)
    pynlpir.open()
    for i, sentence in enumerate(sentences):
        sentence_split_list_entire_sentence = pynlpir.segment(sentence, pos_tagging=False)
        try:
            [sentence_split_list_entire_sentence.index(x) for x in targets[i].strip().split(" ")]
        except ValueError:
            continue
        sentence_reduce = sentence_split_list_entire_sentence
        sentence_clip = sentence_reduce[:]
        targets_list.append(targets[i].strip().split(" "))
        labels_list.append([1, 0] if int(labels[i].strip()) == 0 else [0, 1])
        locations = []
        for j, word in enumerate(sentence_reduce):
            target_location = [sentence_reduce.index(x) for x in targets[i].strip().split(" ")]
            if word not in targets[i].strip().split(" "):
                context_word_relative_location = min([abs(x-j)for x in target_location])
                locations.append(context_word_relative_location)
            else:
                sentence_clip.remove(word)
        sentences_list.append(sentence_clip)
        locations_list.append(locations)
    pynlpir.close()
    wg.finalizeR()
    return sentences_list, targets_list, np.asarray(labels_list), locations_list


def trans_form_location(locations, max_len):
    location_martrix = np.zeros([len(locations), max_len], dtype=np.float32)
    for i, location in enumerate(locations):
        for j, x in enumerate(location):
            location_martrix[i][j] = x
    return location_martrix



def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

