import argparse
import torch
import dill as pickle
from model import LegalQAClassifier
from utils import str2bool
from preprocess import annotate, to_id, init
from train import BatchGen

parser = argparse.ArgumentParser(description='Interact with LegalQA classifier.')
parser.add_argument('--model-file', default='models/best_model.pt')
parser.add_argument("--cuda", type=str2bool, nargs='?',
                    const=True, default=torch.cuda.is_available(),
                    help='whether to use GPU acceleration.')
parser.add_argument('--meta_file', default='coliee_meta_full.msgpack')
args = parser.parse_args()

if args.cuda:
    checkpoint = torch.load(args.model_file)
else:
    checkpoint = torch.load(args.model_file, map_location=lambda storage, loc: storage)

state_dict = checkpoint['state_dict']
opt = checkpoint['config']
with open(args.meta_file, 'rb') as f:
    meta = pickle.load(f)

embedding = torch.Tensor(meta['embedding'])
opt['pretrained_words'] = True
opt['vocab_size'] = embedding.size(0)
opt['embedding_dim'] = embedding.size(1)
opt['pos_size'] = len(meta['vocab_tag'])
opt['ner_size'] = len(meta['vocab_ent'])
opt['cuda'] = args.cuda
opt['classes'] = {'Y': 0, 'N': 1}
opt['id_classes'] = {0: 'Yes', 1: 'No'}
opt['interact'] = True
BatchGen.pos_size = opt['pos_size']
BatchGen.ner_size = opt['ner_size']
model = LegalQAClassifier(opt, embedding, state_dict)
w2id = {w: i for i, w in enumerate(meta['vocab'])}
tag2id = {w: i for i, w in enumerate(meta['vocab_tag'])}
ent2id = {w: i for i, w in enumerate(meta['vocab_ent'])}
init()


def interact_entailment(article, query):
    annotated = annotate(('interact-entailment', article, query, 'Y'))
    model_in = to_id(annotated, w2id, tag2id, ent2id)
    model_in = next(iter(BatchGen([model_in], opt, batch_size=1, gpu=args.cuda, evaluation=True)))
    prediction, values = model.interact_(model_in)

    max_probs = torch.max(prediction, 1)[1].item()
    probs = prediction[0].tolist()
    print('Answer: {0}, with probability Yes: {1:.4f}%, and No: {2:.4f}%'
          .format(opt['id_classes'][max_probs], probs[0] * 100, probs[1] * 100))

    return annotated[1], annotated[5], values # article, query, values


if __name__ == '__main__':
    while True:
        while True:
            article = input('Article: ')
            if article == 'q':
                break
            if article.strip():
                break
        while True:
            query = input('Query: ')
            if query == 'q':
                break

            if query.strip():
                interact_entailment(article, query)