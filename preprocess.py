import collections
import re
import unicodedata

import bcolz
import dill as pickle
import numpy as np

import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_suffix_regex

import os
import logging
import argparse
import multiprocessing
import xml.etree.ElementTree as ET
from functools import partial
from multiprocessing import Pool

dirname, _ = os.path.split(os.path.abspath(__file__))
nlp = None

def main():
    args, log = setup()

    if not args.data_augment:
        train = flatten_xml(args.train_folder, 'train')
        valid = flatten_xml(args.valid_folder, 'valid')
        log.info('xml data flattened.')
    else:
        train, valid = load_augmented_data()
        log.info('xml augmented data loaded.')

    ## tokenize & annotate
    with Pool(args.threads, initializer=init) as p:
        annotate_ = partial(annotate)
        train = list(p.map(annotate_, train, chunksize=args.batch_size))
        valid = list(p.imap(annotate_, valid, chunksize=args.batch_size))

    initial_len = len(train)
    train = list(filter(lambda x: x[-1] is not None, train))
    log.info('drop {} inconsistent samples.'.format(initial_len - len(train)))
    log.info('tokens generated')

    full = train + valid
    t1 = [row[1] for row in full]
    t2 = [row[5] for row in full]

    # build vocabulary
    vocab, counter = build_vocab(t1, t2)

    counter_tag = collections.Counter(w for row in full for w in row[3])
    vocab_tag = sorted(counter_tag, key=counter_tag.get, reverse=True)
    counter_ent = collections.Counter(w for row in full for w in row[4])
    vocab_ent = sorted(counter_ent, key=counter_ent.get, reverse=True)
    w2id = {w: i for i, w in enumerate(vocab)}
    id2w = {i: w for i, w in enumerate(vocab)}
    tag2id = {w: i for i, w in enumerate(vocab_tag)}
    ent2id = {w: i for i, w in enumerate(vocab_ent)}
    log.info('Vocabulary size: {}'.format(len(vocab)))
    log.info('Found {} POS tags.'.format(len(vocab_tag)))
    log.info('Found {} entity tags: {}'.format(len(vocab_ent), vocab_ent))

    """
    pair_id     = row[0]
    t1_tokens   = row[1]
    t1_features = row[2] 
    t1_tags     = row[3] 
    t1_ents     = row[4] 
    t2_tokens   = row[5]
    t1          = row[6] # original t1 text
    t2          = row[7] # original t2 text
    label       = row[8] # string label Y/N
    """

    to_id_ = partial(to_id, w2id=w2id, tag2id=tag2id, ent2id=ent2id)
    train = list(map(to_id_, train))
    valid = list(map(to_id_, valid))
    log.info('converted to ids.')

    # loading glove
    glove_dir = os.path.dirname(args.wv_file)
    vectors_path  = os.path.join(glove_dir, 'glove.840B.300d.dat')
    words_path    = os.path.join(glove_dir, 'glove.840B.300d_words.pkl')
    word2idx_path = os.path.join(glove_dir, 'glove.840B.300d_idx.pkl')
    if not os.path.exists(words_path):
        build_glove(args.wv_file)
        log.info('glove built.')

    vectors  = bcolz.open(vectors_path)[:]
    words    = pickle.load(open(words_path, 'rb'))
    word2idx = pickle.load(open(word2idx_path, 'rb'))
    glove    = {w: vectors[word2idx[w]] for w in words}
    log.info('glove loaded.')

    vocab_size = len(vocab)
    embeddings = np.zeros((vocab_size, args.wv_dim))
    embed_counts = np.zeros(vocab_size)
    embed_counts[:4] = 1  # PAD, SOS, EOS, UNK

    words_found = 0
    for i, word in enumerate(w2id):
        if word in ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]:
            continue
        try:
            embeddings[i] = glove[word]
            words_found += 1
        except KeyError:
            embeddings[i] = np.random.normal(scale=0.6, size=(args.wv_dim, ))

        embed_counts[i] += 1

    embeddings /= embed_counts.reshape((-1, 1))
    log.info('got embedding matrix.')
    log.info('{0} words not found.'.format(vocab_size - words_found))

    meta = {
        'vocab': vocab,
        'vocab_tag': vocab_tag,
        'vocab_ent': vocab_ent,
        'embedding': embeddings.tolist(),
        'word2id': w2id,
        'id2word': id2w,
        'wv_cased': args.wv_cased,
    }
    with open('data/coliee_meta_full_ko-de-en.msgpack', 'wb') as f:
        pickle.dump(meta, f)

    result = {
        'train': train,
        'valid': valid
    }
    with open('data/coliee_data_full_ko-de-en.msgpack', 'wb') as f:
        pickle.dump(result, f)

    log.info('saved to disk.')

def to_id(row, w2id, tag2id, ent2id):
    t1_tokens = row[1]
    t1_features = row[2]
    t1_tags = row[3]
    t1_ents = row[4]
    t2_tokens = row[5]

    t1_ids = [w2id[w] for w in t1_tokens]
    t2_ids = [w2id[w] for w in t2_tokens]

    tag_ids = [tag2id[w] for w in t1_tags]
    ent_ids = [ent2id[w] for w in t1_ents]

    assert len(t1_ids) >= len(tag_ids)
    assert len(t1_ids) >= len(ent_ids)

    return (row[0], t1_ids, t1_features, tag_ids, ent_ids, t2_ids) + row[6:]

def init():
    """initialize spacy in each process"""
    global nlp
    nlp = spacy.load('en')
    nlp.tokenizer = custom_tokenizer(nlp)

def build_vocab(t1, t2):
    """
    Build vocabulary
    """
    t1_vocab = [w for doc in t1 for w in doc]
    t2_vocab = [w for doc in t2 for w in doc]

    counter_t1 = collections.Counter(t1_vocab)
    counter_t2 = collections.Counter(t2_vocab)
    counter = counter_t2 + counter_t1
    vocab = sorted([t for t in counter_t1], key=counter_t1.get, reverse=True)
    vocab += sorted([t for t in counter_t2.keys() - counter_t1.keys()], key=counter.get, reverse=True)

    vocab.insert(0, "<PAD>")
    vocab.insert(1, "<SOS>")
    vocab.insert(2, "<EOS>")
    vocab.insert(3, "<UNK>")
    return vocab, counter

def build_glove(glove_file):
    words = [] # number of vocabularies
    idx = 0
    word2idx = {}
    glove_dir = os.path.dirname(glove_file)

    vectors_path  = os.path.join(glove_dir, 'glove.840B.300d.dat')
    words_path    = os.path.join(glove_dir, 'glove.840B.300d_words.pkl')
    word2idx_path = os.path.join(glove_dir, 'glove.840B.300d_idx.pkl')
    vectors = bcolz.carray(np.zeros(1), rootdir=vectors_path, mode='w')
    with open(glove_file) as f:
        for line in f:
            elems = line.rstrip().split(' ')
            word = unicode_to_ascii(elems[0])
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(elems[1:]).astype(np.float)
            vectors.append(vect)

    vocab_size = len(words)
    vectors = bcolz.carray(vectors[1:].reshape((vocab_size, 300)), rootdir=vectors_path, mode='w')
    vectors.flush()
    pickle.dump(words, open(words_path, 'wb'))
    pickle.dump(word2idx, open(word2idx_path, 'wb'))

def annotate(row):
    global nlp

    id_, t1, t2 = row[:3]
    t1_doc = nlp(clean_spaces(t1))
    t2_doc = nlp(clean_spaces(t2))

    t1_tokens = [unicode_to_ascii(token.orth_) for token in t1_doc if not token.orth_.isspace()]
    t2_tokens = [unicode_to_ascii(token.orth_) for token in t2_doc if not token.orth_.isspace()]

    t1_tokens_lower = [w.lower() for w in t1_tokens]
    t2_tokens_lower = [w.lower() for w in t2_tokens]

    t1_tags = [w.tag_ for w in t1_doc]
    t1_ents = [w.ent_type_ for w in t1_doc]

    assert len(t1_tokens) == len(t1_doc)
    assert len(t1_tokens) == len(t1_tags)
    assert len(t1_tokens) == len(t1_ents)
    assert len(t1_tokens) == len(t1_tokens_lower)

    t2_lemma = {w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower() for w in t2_doc}
    t2_tokens_set = set(t2_tokens)
    t2_tokens_lower_set = set(t2_tokens_lower)
    match_origin = [w in t2_tokens_set for w in t1_tokens]
    match_lower = [w in t2_tokens_lower_set for w in t1_tokens_lower]
    match_lemma = [(w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower()) in t2_lemma for w in t1_doc]
    # term frequency in document
    counter_ = collections.Counter(t1_tokens_lower)
    total = len(t1_tokens_lower)
    t1_tf = [counter_[w] / total for w in t1_tokens_lower]
    t1_features = list(zip(match_origin, match_lower, match_lemma, t1_tf))

    t1_tokens = t1_tokens_lower
    t2_tokens = t2_tokens_lower

    return (id_, t1_tokens, t1_features, t1_tags, t1_ents,
            t2_tokens, t1, t2) + row[3:]

def load_augmented_data():
    # languages = "orig de fr la da nl ko es pt el".split()
    languages = "orig ko-de-en".split()

    train = []
    valid = []

    for lang in languages:
        with open('data/coliee_data_{0}.msgpack'.format(lang), 'rb') as f:
            data = pickle.load(f)

        train.extend(data['train'])
        valid.extend(data['valid'])

    # for lang in languages:
    #     with open('data/coliee_data_{0}.msgpack'.format(lang), 'rb') as f:
    #         data = pickle.load(f)
    #
    #     train.extend(data['train'])
    #     valid.extend(data['valid'])

    # random.shuffle(train)
    # random.shuffle(train)

    return train, valid

def flatten_xml(folder, mode):
    rows = []
    if mode == 'train':
        xml_file_prefix = 'riteval_H'

        for files in range(18, 28):
            fname = "{0}/{1}{2}.xml".format(folder, xml_file_prefix, files)
            # fname = os.path.join(dirname, fname) # in case full path needed
            with open(fname, 'r') as f:
                root = ET.fromstring(f.read())

            for child in root:
                id_ = child.attrib["id"]
                label = child.attrib["label"]
                t1 = child[0].text
                t2 = child[1].text

                t1_orig = " ".join(t1.split("\n")).lstrip().rstrip()  # removing "\n"
                t2_orig = " ".join(t2.split("\n")).lstrip().rstrip()  # removing "\n"
                rows.append((id_, t1_orig, t2_orig, label))

            print("processed: {0}".format(fname))

    if mode == "valid":
        fname = "{0}/riteval_H28.xml".format(folder)
        # fname = os.path.join(dirname, fname)     # in case full path needed
        with open(fname, 'r') as f:
            root = ET.fromstring(f.read())

        for child in root:
            id_ = child.attrib["id"]
            label = child.attrib["label"]
            t1 = child[0].text
            t2 = child[1].text

            t1_orig = " ".join(t1.split("\n")).lstrip().rstrip()  # removing "\n"
            t2_orig = " ".join(t2.split("\n")).lstrip().rstrip()  # removing "\n"
            rows.append((id_, t1_orig, t2_orig, label))

        print("processed: {0}".format(fname))

    return rows

def clean_spaces(text):
    """normalize with spaces in a string."""

    remap = {
        ord('\t'): ' ',
        ord('\f'): ' ',
        ord('\r'): None
    }
    space_norm = re.sub('\s+', ' ', text.translate(remap))
    return space_norm

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def setup():

    parser = argparse.ArgumentParser(
        description='Pre-processing data files!'
    )
    parser.add_argument('--train_folder', default='data/coliee/train',
                        help='path to train folder.')
    parser.add_argument('--valid_folder', default='data/coliee/valid',
                        help='path to valid folder.')
    parser.add_argument('--wv_file', default='data/glove/glove.840B.300d.txt',
                        help='path to word vector file.')
    parser.add_argument('--wv_dim', type=int, default=300,
                        help='word vector dimension.')
    parser.add_argument('--wv_cased', type=str2bool, nargs='?',
                        const=True, default=True,
                        help='treat the words as cased or not.')
    parser.add_argument('--sort_all', action='store_true',
                        help='sort the vocabulary by frequencies of all words. '
                             'Otherwise consider question words first.')
    parser.add_argument('--sample_size', type=int, default=0,
                        help='size of sample data (for debugging).')
    parser.add_argument('--threads', type=int, default=min(multiprocessing.cpu_count(), 16),
                        help='number of threads for preprocessing.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for multiprocess tokenizing and tagging.')
    parser.add_argument('--data_augment', type=str2bool, nargs='?',
                        const=True, default=True,
                        help='use data augment or not.')

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG,
                        datefmt='%m/%d/%Y %I:%M:%S')
    log = logging.getLogger(__name__)
    log.info(vars(args))
    log.info('start data preparing...')

    return args, log

def custom_tokenizer(nlp):
    infix_re = re.compile(r'''[.\,\(\)\?\:\;\...\‘\’\`\“\”\"\'~]''')
    prefix_re = compile_prefix_regex(nlp.Defaults.prefixes)
    suffix_re = compile_suffix_regex(nlp.Defaults.suffixes)

    return Tokenizer(nlp.vocab, prefix_search=prefix_re.search,
                                suffix_search=suffix_re.search,
                                infix_finditer=infix_re.finditer,
                                token_match=None)

if __name__ == '__main__':
    main()