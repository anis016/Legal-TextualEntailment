import os
from datetime import datetime
import random
from shutil import copyfile

import dill as pickle
from argparse import ArgumentParser

import torch
from torch.nn.modules.module import _addindent

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from utils import str2bool
from model import LegalQAClassifier

total_params = 0

def torch_summarize(model, show_weights=True, show_parameters=True):
    global total_params

    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    try:
        model_dir = model.network._modules.items()
    except:
        model_dir = model._modules.items()

    for key, module in model_dir:
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__str__()

        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        if type(model) != torch.nn.modules.container.Sequential:
            total_params += params

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr

def load_data(opt):

    with open(opt['meta_file'], 'rb') as f:
        meta = pickle.load(f)

    embedding = torch.Tensor(meta['embedding'])
    opt['id2word'] = meta['id2word']
    opt['pretrained_words'] = True
    opt['vocab_size'] = embedding.size(0)
    opt['embedding_dim'] = embedding.size(1)
    opt['pos_size'] = len(meta['vocab_tag'])
    opt['ner_size'] = len(meta['vocab_ent'])
    BatchGen.pos_size = opt['pos_size']
    BatchGen.ner_size = opt['ner_size']

    with open(opt['data_file'], 'rb') as f:
        data = pickle.load(f)

    train = data['train']
    data['valid'].sort(key=lambda x: len(x[1]))
    valid = data['valid']

    opt['classes'] = {'Y': 0, 'N': 1}
    return train, valid, embedding, opt

def get_args():
    parser = ArgumentParser(description='PyTorch/Legal QA.')
    parser.add_argument('--data_file', default='coliee_data_full.msgpack')
    parser.add_argument('--meta_file', default='coliee_meta_full.msgpack')
    # parser.add_argument('--data_file', default='coliee_data_full_new.msgpack')
    # parser.add_argument('--meta_file', default='coliee_meta_full_new.msgpack')
    # parser.add_argument('--data_file', default='coliee_data_full_trans.msgpack')
    # parser.add_argument('--meta_file', default='coliee_meta_full_trans.msgpack')
    # parser.add_argument('--data_file', default='coliee_data_full_ko-de-en.msgpack')
    # parser.add_argument('--meta_file', default='coliee_meta_full_ko-de-en.msgpack')
    parser.add_argument('--model_dir', default='models',
                        help='path to store saved models.')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--t1_layers', type=int, default=3)
    parser.add_argument('--t2_layers', type=int, default=3)
    parser.add_argument('--encoder_layer', type=int, default=1)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--no-bidirectional', action='store_false', dest='birnn')
    parser.add_argument('--preserve-case', action='store_false', dest='lower')
    parser.add_argument('--no-projection', action='store_false', dest='projection')
    parser.add_argument('--train_embed', action='store_false', dest='fix_emb')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--concat_rnn_layers', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--dropout_rnn_output', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--dropout_emb', type=float, default=0.2)
    parser.add_argument('--dropout_rnn', type=float, default=0.2)
    parser.add_argument('--dropout_linear', type=int, default=0.1)
    parser.add_argument('--num_features', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1013)
    parser.add_argument('-gc', '--grad_clipping', type=float, default=10)
    parser.add_argument("--cuda", type=str2bool, nargs='?',
                        const=True, default=torch.cuda.is_available())
    parser.add_argument('-op', '--optimizer', default='adamax',
                        help='supported optimizer: adamax, sgd')
    parser.add_argument('--fix_embeddings', action='store_true',
                        help='if true, `tune_partial` will be ignored.')
    parser.add_argument('--use_t2_emb', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--pos', type=str2bool, nargs='?', const=False, default=False)
    parser.add_argument('--ner', type=str2bool, nargs='?', const=False, default=False)
    parser.add_argument('--rnn_type', default='lstm',
                        help='supported types: rnn, gru, lstm')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0)
    parser.add_argument('--log_per_updates', type=int, default=20)
    parser.add_argument('--interact', type=str2bool, nargs='?', const=False, default=False)

    args = parser.parse_args()

    # set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    return args

class BatchGen:
    pos_size = None
    ner_size = None

    def __init__(self, data, options, batch_size, gpu, evaluation=False):
        """
        input:
            data - list of lists
            batch_size - int
        """
        self.batch_size = batch_size
        self.eval = evaluation
        self.gpu = gpu
        self.options = options

        # sort by len
        data = sorted(data, key=lambda x: len(x[1]))
        # chunk into batches
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

        # shuffle
        if not evaluation:
            random.shuffle(data)

        self.data = data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for batch in self.data:
            batch_size = len(batch)
            batch = list(zip(*batch))

            assert len(batch) == 9

            t1_len = max(len(x) for x in batch[1])
            t1_id = torch.LongTensor(batch_size, t1_len).fill_(0)
            for i, doc in enumerate(batch[1]):
                t1_id[i, :len(doc)] = torch.LongTensor(doc)

            feature_len = len(batch[2][0][0])

            t1_feature = torch.Tensor(batch_size, t1_len, feature_len).fill_(0)
            for i, doc in enumerate(batch[2]):
                for j, feature in enumerate(doc):
                    t1_feature[i, j, :] = torch.Tensor(feature)

            t1_tag = torch.Tensor(batch_size, t1_len, self.pos_size).fill_(0)
            for i, doc in enumerate(batch[3]):
                for j, tag in enumerate(doc):
                    t1_tag[i, j, tag] = 1

            t1_ent = torch.Tensor(batch_size, t1_len, self.ner_size).fill_(0)
            for i, doc in enumerate(batch[4]):
                for j, ent in enumerate(doc):
                    t1_ent[i, j, ent] = 1

            t2_len = max(len(x) for x in batch[5])
            t2_id = torch.LongTensor(batch_size, t2_len).fill_(0)
            for i, doc in enumerate(batch[5]):
                t2_id[i, :len(doc)] = torch.LongTensor(doc)

            t1_mask = torch.eq(t1_id, 0)
            t2_mask = torch.eq(t2_id, 0)

            label = []
            for ix in range(len(batch[8])):
                label.append(self.options['classes'][batch[8][ix]])
            label = torch.LongTensor(label)

            if self.gpu:
                t1_id = t1_id.pin_memory()
                t1_feature = t1_feature.pin_memory()
                t1_tag = t1_tag.pin_memory()
                t1_ent = t1_ent.pin_memory()
                t1_mask = t1_mask.pin_memory()
                t2_id = t2_id.pin_memory()
                t2_mask = t2_mask.pin_memory()
                label = label.pin_memory()

            yield (t1_id, t1_feature, t1_tag, t1_ent, t1_mask,
                   t2_id, t2_mask, label)

def main():
    args = get_args()
    torch.cuda.set_device(args.gpu)
    train_, valid_, embedding, options = load_data(vars(args))

    config = args
    config.n_embed = options['vocab_size']
    config.d_out = len(options['classes'])

    model = LegalQAClassifier(options, embedding)
    print(model.network)
    torch_summarize(model)
    print("total parameters: ", total_params)

    best_val_score = 0.0
    for epoch in range(args.epochs + 1):
        y_pred = []
        y_true = []

        batches = BatchGen(train_, options=options, batch_size=args.batch_size, gpu=args.cuda)
        start = datetime.now()
        for batch_idx, batch in enumerate(batches):

            model.update(batch)

            # if batch_idx % options['log_per_updates'] == 0:
            #     print('> epoch [{0:2}] updates[{1:6}] train loss[{2:.5f}] remaining[{3}]'.format(
            #         epoch, model.updates, model.train_loss.value,
            #         str((datetime.now() - start) / (batch_idx + 1) * (len(batches) - batch_idx - 1)).split('.')[0]))

        batches = BatchGen(valid_, options=options, batch_size=args.batch_size, evaluation=True, gpu=args.cuda)
        for batch_idx, batch in enumerate(batches):
            prediction = model.predict(batch)

            y_pred.extend(prediction.tolist())
            y_true.extend(batch[7].tolist())

        val_acc = accuracy_score(y_true, y_pred)
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        print("Epoch[{0:3}] Precision[{1:.4f}], Recall[{2:.4f}], F1-Measure[{3:.4f}], Val Accuracy[{4:.4f}]".format(epoch + 1, precision, recall, f1_score, val_acc))

        # save
        if val_acc > best_val_score and epoch > 0:
            best_val_score = val_acc
            model_file = os.path.join(args.model_dir, 'checkpoint_epoch_{0}_val_{1}.pt'.format(epoch, val_acc))

            model.save(model_file, epoch, [precision, recall, f1_score, val_acc])
            copyfile(
                model_file,
                os.path.join(args.model_dir, 'best_model.pt'))
            print('[new best model saved.]')

if __name__ == '__main__':
    main()