import os
import ast
import spacy
import numpy as np
import xml.etree.ElementTree as ET
from errno import ENOENT
from collections import Counter


nlp = spacy.load("en")


def get_data_info(train_fname, test_fname, save_fname, pre_processed):
    word2id, word2id_entity_attribute, max_sentence_len, max_aspect_entity_len, max_aspect_attribute_len = {}, {}, 0, 0, 0
    word2id['<pad>'] = 0
    word2id_entity_attribute['<pad>'] = 0
    if pre_processed:
        if not os.path.isfile(save_fname):
            raise IOError(ENOENT, 'Not a file', save_fname)
        with open(save_fname, 'r') as f:
            for line in f:
                content = line.strip().split()
                if len(content) == 4:
                    max_sentence_len = int(content[1])
                    max_aspect_entity_len = int(content[2])
                    max_aspect_attribute_len = int(content[3])
                elif len(content) == 3:
                    word2id_entity_attribute[content[0]] = int(content[1])
                else:
                    word2id[content[0]] = int(content[1])
    else:
        if not os.path.isfile(train_fname):
            raise IOError(ENOENT, 'Not a file', train_fname)
        if not os.path.isfile(test_fname):
            raise IOError(ENOENT, 'Not a file', test_fname)

        words = []

        train_tree = ET.parse(train_fname)
        train_root = train_tree.getroot()
        for review in train_root:
            for sentences_in_review in review:
                for sentence in sentences_in_review:
                    sptoks = nlp(sentence.find('text').text)
                    words.extend([sp.text.lower() for sp in sptoks])
                    if len(sptoks) > max_sentence_len:
                        max_sentence_len = len(sptoks)
                    for asp_terms in sentence.iter('Opinions'):
                        for asp_term in asp_terms.findall('Opinion'):
                            if asp_term.get('polarity') == 'conflict':
                                continue
                            if asp_term.get('category') == 'NULL':
                                continue
                            t_sptoks_0 = asp_term.get('category').split("#")[0]
                            t_sptoks_1 = asp_term.get('category').split("#")[1]
                            if t_sptoks_0 not in word2id_entity_attribute:
                                word2id_entity_attribute[t_sptoks_0] = len(word2id_entity_attribute)
                            if t_sptoks_1 not in word2id_entity_attribute:
                                word2id_entity_attribute[t_sptoks_1] = len(word2id_entity_attribute)

        word_count = Counter(words).most_common()
        for word, _ in word_count:
            if word not in word2id and ' ' not in word:
                word2id[word] = len(word2id)
    
        test_tree = ET.parse(test_fname)
        test_root = test_tree.getroot()
        for review in test_root:
            for sentences_in_review in review:
                for sentence in sentences_in_review:
                    sptoks = nlp(sentence.find('text').text)
                    words.extend([sp.text.lower() for sp in sptoks])
                    if len(sptoks) > max_sentence_len:
                        max_sentence_len = len(sptoks)
                    for asp_terms in sentence.iter('Opinions'):
                        for asp_term in asp_terms.findall('Opinion'):
                            if asp_term.get('polarity') == 'conflict':
                                continue
                            if asp_term.get('category') == 'NULL':
                                continue
                            t_sptoks_0 = asp_term.get('category').split("#")[0]
                            t_sptoks_1 = asp_term.get('category').split("#")[1]
                            if t_sptoks_0 not in word2id_entity_attribute:
                                word2id_entity_attribute[t_sptoks_0] = len(word2id_entity_attribute)
                            if t_sptoks_1 not in word2id_entity_attribute:
                                word2id_entity_attribute[t_sptoks_1] = len(word2id_entity_attribute)


        word_count = Counter(words).most_common()
        for word, _ in word_count:
            if word not in word2id and ' ' not in word:
                word2id[word] = len(word2id)

        max_aspect_entity_len = 1
        max_aspect_attribute_len = 1

        with open(save_fname, 'w') as f:
            f.write('length %s %s %s\n' % (max_sentence_len, max_aspect_entity_len, max_aspect_attribute_len))
            for key, value in word2id.items():
                f.write('%s %s\n' % (key, value))
            for key, value in word2id_entity_attribute.items():
                f.write('entity_attribute %s %s\n' % (key, value))
                
    print('There are %s words in the dataset, %s words in the category,  the max length of sentence is %s, and the max length of aspect_entity is %s and aspect_attribute is %s' % (len(word2id), len(word2id_entity_attribute), max_sentence_len, max_aspect_entity_len, max_aspect_attribute_len))
    return word2id, word2id_entity_attribute, max_sentence_len, max_aspect_entity_len, max_aspect_attribute_len

def get_loc_info(sptoks, from_id, to_id):
    aspect = []
    for sptok in sptoks:
        if sptok.idx < to_id and sptok.idx + len(sptok.text) > from_id:
            aspect.append(sptok.i)
    loc_info = []
    for _i, sptok in enumerate(sptoks):
        loc_info.append(min([abs(_i - i) for i in aspect]) / len(sptoks))
    return loc_info

def read_data(fname, word2id, word2id_entity_attribute, max_sentence_len, save_fname, pre_processed):
    sentences, aspects_entity, aspects_attribute, sentence_lens, labels = [], [], [], [], []
    if pre_processed:
        if not os.path.isfile(save_fname):
            raise IOError(ENOENT, 'Not a file', save_fname)
        lines = open(save_fname, 'r').readlines()
        for i in range(0, len(lines), 5):
            sentences.append(ast.literal_eval(lines[i]))
            aspects_entity.append(ast.literal_eval(lines[i + 1]))
            aspects_attribute.append(ast.literal_eval(lines[i + 2]))
            sentence_lens.append(ast.literal_eval(lines[i + 3]))
            labels.append(ast.literal_eval(lines[i + 4]))
    else:
        if not os.path.isfile(fname):
            raise IOError(ENOENT, 'Not a file', fname)

        tree = ET.parse(fname)
        root = tree.getroot()
        with open(save_fname, 'w') as f:
            for review in root:
                for sentences_in_review in review:
                    for sentence in sentences_in_review:
                        sptoks = nlp(sentence.find('text').text)
                        if len(sptoks.text.strip()) != 0:
                            ids = []
                            for sptok in sptoks:
                                if sptok.text.lower() in word2id:
                                    ids.append(word2id[sptok.text.lower()])
                            for asp_terms in sentence.iter('Opinions'):
                                for asp_term in asp_terms.findall('Opinion'):
                                    if asp_term.get('polarity') == 'conflict':
                                        continue
                                    if asp_term.get('category') == 'NULL':
                                        continue
                                    t_sptoks_0 = asp_term.get('category').split("#")[0]
                                    t_sptoks_1 = asp_term.get('category').split("#")[1]
                                    # t_ids = []
                                    # for sptok in t_sptoks_0:
                                    #     if sptok.text.lower() in word2id:
                                    #         t_ids.append(word2id[sptok.text.lower()])
                                    sentences.append(ids + [0] * (max_sentence_len - len(ids)))
                                    f.write("%s\n" % sentences[-1])
                                    aspects_entity.append(word2id_entity_attribute[t_sptoks_0])
                                    f.write("%s\n" % aspects_entity[-1])
                                    aspects_attribute.append(word2id_entity_attribute[t_sptoks_1])
                                    f.write("%s\n" % aspects_attribute[-1])
                                    sentence_lens.append(len(sptoks))
                                    f.write("%s\n" % sentence_lens[-1])
                                    # loc_info = get_loc_info(sptoks, int(asp_term.get('from')), int(asp_term.get('to')))
                                    # sentence_locs.append(loc_info + [1] * (max_sentence_len - len(loc_info)))
                                    # f.write("%s\n" % sentence_locs[-1])
                                    polarity = asp_term.get('polarity')
                                    if polarity == 'negative':
                                        labels.append([1, 0, 0])
                                    elif polarity == 'neutral':
                                        labels.append([0, 1, 0])
                                    elif polarity == "positive":
                                        labels.append([0, 0, 1])
                                    f.write("%s\n" % labels[-1])

    print("Read %s sentences from %s" % (len(sentences), fname))
    # return np.asarray(sentences), np.asarray(aspects), np.asarray(sentence_lens), np.asarray(sentence_locs), np.asarray(labels)
    return np.asarray(sentences), np.reshape(np.asarray(aspects_entity), (-1, 1)), np.reshape(np.asarray(aspects_attribute), (-1, 1)), np.asarray(labels)

def load_word_embeddings(fname, embedding_dim, word2id):
    if not os.path.isfile(fname):
        raise IOError(ENOENT, 'Not a file', fname)

    word2vec = np.random.normal(0, 0.05, [len(word2id), embedding_dim])
    oov = len(word2id)
    with open(fname, 'rb') as f:
        for line in f:
            line = line.decode('utf-8')
            content = line.strip().split()
            if '.' in content:
                number = content.count('.')
                word = ['' * number]
                word.extend(content[number:])
                content = word
            if 'name@domain.com' in content:
                index = content.index('name@domain.com')
                index += 1
                a = ''
                for i in range(index):
                    a += ' ' + content[i]
                word = [a.strip()]
                word.extend(content[index:])
                content = word
            if 'mylot.com' in content:
                index = content.index('mylot.com')
                index += 1
                a = ''
                for i in range(index):
                    a += ' ' + content[i]
                word = [a.strip()]
                word.extend(content[index:])
                content = word
            if 'Amazon.com' in content:
                index = content.index('Amazon.com')
                index += 1
                a = ''
                for i in range(index):
                    a += ' ' + content[i]
                word = [a.strip()]
                word.extend(content[index:])
                content = word
            if content[0] in word2id:
                word2vec[word2id[content[0]]] = np.array(list(map(float, content[1:])))
                oov = oov - 1
    word2vec[word2id['<pad>'], :] = 0
    print('There are %s words in vocabulary and %s words out of vocabulary' % (len(word2id) - oov, oov))
    return word2vec

# def get_batch_index(length, batch_size, is_shuffle=True):
#     index = list(range(length))
#     if is_shuffle:
#         np.random.shuffle(index)
#     for i in range(int(length / batch_size) + (1 if length % batch_size else 0)):
#         yield index[i * batch_size:(i + 1) * batch_size]


def get_batch_index(length, batch_size, is_shuffle=True):
    index = list(range(length))
    if is_shuffle:
        np.random.shuffle(index)
    for i in range(int(length / batch_size)):
        yield index[i * batch_size:(i + 1) * batch_size]

def statics_polarity(labels):
    pos = 0
    neg = 0
    neu = 0
    for lable in labels:
        if lable == 2:
            pos += 1
        if lable == 0:
            neg += 1
        if lable == 1:
            neu += 1
    return pos, neg, neu