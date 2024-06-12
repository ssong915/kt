import torch, sys
#  from torchtext.data.functional import to_map_style_dataset

import numpy as np
#  from collections import defaultdict


def get_dataloaders(data_dict, batch_size, device, ns_method, label='Train'):

    if label == 'Train':
        train_pos_data = data_dict["train_only_pos"] + data_dict["ground_train"]
        train_pos_labels = [1 for i in range(len(train_pos_data))] # training positive hyperedges
        train_pos_dataloader = BatchDataloader(train_pos_data, train_pos_labels, batch_size, device, is_Train=True)

        train_neg_data = data_dict["train_sns"]
        train_neg_labels = [0 for i in range(len(train_neg_data))] # training negative hyperedges
        train_neg_dataloader = BatchDataloader(train_neg_data, train_neg_labels, batch_size, device, is_Train=True)

        return train_pos_dataloader, train_neg_dataloader

    elif label == 'Valid':
        val_pos_data = data_dict["valid_only_pos"] + data_dict["ground_valid"]
        val_pos_labels = [1 for i in range(len(val_pos_data))] # validation positive hyperedges
        val_pos_dataloader = BatchDataloader(val_pos_data, val_pos_labels, batch_size, device, is_Train=False)

        val_neg_data = data_dict["valid_sns"]
        val_neg_labels = [0 for i in range(len(val_neg_data))] # validation negative hyperedges
        val_neg_dataloader = BatchDataloader(val_neg_data, val_neg_labels, batch_size, device, is_Train=False)

        return val_pos_dataloader, val_neg_dataloader

    elif label == 'Test':
        test_pos_data = data_dict["test_pos"]
        test_pos_labels = [1 for i in range(len(test_pos_data))] # validation positive hyperedges
        test_pos_dataloader = BatchDataloader(test_pos_data, test_pos_labels, batch_size, device, is_Train=False)

        test_neg_data = data_dict["test_sns"]
        test_neg_labels = [0 for i in range(len(test_neg_data))] # validation negative hyperedges
        test_neg_dataloader = BatchDataloader(test_neg_data, test_neg_labels, batch_size, device, is_Train=False)

        return test_pos_dataloader, test_neg_dataloader
    else:
        sys.exit('Invalid data labels: Train, Eval, Test')



class BatchDataloader(object):
    def __init__(self, hyperedges, labels, batch_size, device, is_Train=False):
        """Creates an instance of Hyperedge Batch Dataloader.
        Args:
            hyperedges: List(frozenset). List of hyperedges.
            labels: list. Labels of hyperedges.
            batch_size. int. Batch size of each batch.
        """
        self.batch_size = batch_size
        self.hyperedges = hyperedges
        self.labels = labels
        self._cursor = 0
        self.device = device
        self.is_Train = is_Train

        if is_Train:
            self.shuffle()

    def eval(self):
        self.test_generator = True

    def train(self):
        self.test_generator = False

    def shuffle(self):
        idcs = np.arange(len(self.hyperedges))
        np.random.shuffle(idcs)
        self.hyperedges = [self.hyperedges[i] for i in idcs]
        self.labels = [self.labels[i] for i in idcs]

    def __iter__(self):
        self._cursor = 0
        return self

    def next(self):
        return self._next_batch()

    def _next_batch(self):
        ncursor = self._cursor+self.batch_size # next cursor position

        next_hyperedges = None
        next_labels = None

        if ncursor >= len(self.hyperedges): # end of each epoch
            next_hyperedges = self.hyperedges[self._cursor:]
            next_labels = self.labels[self._cursor:]
            self._cursor = 0

            if self.is_Train:
                self.shuffle() # data shuffling at every epoch

        else:
            next_hyperedges = self.hyperedges[self._cursor:self._cursor + self.batch_size]
            next_labels = self.labels[self._cursor:self._cursor + self.batch_size]
            self._cursor = ncursor % len(self.hyperedges)

        hyperedges = next_hyperedges[:]
        labels = torch.FloatTensor(next_labels).to(self.device)

        return hyperedges, labels


