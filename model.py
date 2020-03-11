import random

import torch
import torch.nn as nn
from rnn_reader import ArticleReader
from torch.autograd import Variable
from utils import AverageMeter

import logging
logger = logging.getLogger(__name__)

class LegalQAClassifier(object):

    def __init__(self, opt, embedding=None, state_dict=None):
        super(LegalQAClassifier, self).__init__()

        self.opt = opt
        self.device = torch.cuda.current_device() if opt['cuda'] else torch.device('cpu')
        self.updates = state_dict['updates'] if state_dict else 0
        self.train_loss = AverageMeter()
        if state_dict:
            self.train_loss.load(state_dict['loss'])

        # Building network.
        self.network = ArticleReader(opt, embedding=embedding)
        if state_dict:
            new_state = set(self.network.state_dict().keys())
            for k in list(state_dict['network'].keys()):
                if k not in new_state:
                    del state_dict['network'][k]
            self.network.load_state_dict(state_dict['network'])
        self.network.to(self.device)

        # Building optimizer.
        self.opt_state_dict = state_dict['optimizer'] if state_dict else None
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(parameters, lr=self.opt['lr'])

        if self.opt_state_dict:
            self.optimizer.load_state_dict(self.opt_state_dict)

        # loss
        self.criterion = nn.CrossEntropyLoss()

    def update(self, ex):
        # Train mode
        self.network.train()

        # Transfer to GPU
        inputs = [e.to(self.device) for e in ex[:7]]
        target = ex[7].to(self.device)

        # Run forward
        answer = self.network(*inputs)

        # Compute loss and accuracies
        loss = self.criterion(answer, target)
        self.train_loss.update(loss.item())

        # Clear gradients and run backward
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.network.parameters(),
                                       self.opt['grad_clipping'],
                                       norm_type=2)

        # Update parameters
        self.optimizer.step()
        self.updates += 1

    def predict(self, ex):
        # Eval mode
        self.network.eval()

        # Transfer to GPU
        if self.opt['cuda']:
            inputs = [Variable(e.cuda(async=True)) for e in ex[:7]]
        else:
            inputs = [Variable(e) for e in ex[:7]]

        # Run forward
        with torch.no_grad():
            answer = self.network(*inputs)

        # Transfer to CPU/normal tensors for numpy ops
        answer = answer.data.cpu()
        answer = torch.max(answer, 1)[1]

        return answer

    def interact_(self, ex):
        # Eval mode
        self.network.eval()

        # Transfer to GPU
        if self.opt['cuda']:
            inputs = [Variable(e.cuda(async=True)) for e in ex[:7]]
        else:
            inputs = [Variable(e) for e in ex[:7]]

        # Run forward
        with torch.no_grad():
            answer, values = self.network(*inputs)

        # Transfer to CPU/normal tensors for numpy ops
        answer = answer.data.cpu()
        answer = torch.nn.functional.sigmoid(answer)

        return answer, values

    def save(self, filename, epoch, score):
        precision, recall, f1, valid_acc = score
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'updates': self.updates,
                'loss': self.train_loss.state_dict()
            },
            'config': self.opt,
            'epoch': epoch,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'best_valid': valid_acc,
            'random_state': random.getstate(),
            'torch_state': torch.random.get_rng_state(),
            'torch_cuda_state': torch.cuda.get_rng_state()
        }

        try:
            torch.save(params, filename)
            logger.info('model saved to {}'.format(filename))
        except BaseException:
            logger.warning('[ WARN: Saving failed... continuing anyway. ]')