import torch
import multiprocessing
import numpy as np
import kaldi_io
from tdataset import TensorDataset
from torch.utils.data import DataLoader

def _fetch_caches(datastream, index_queue, data_queue, labels, batchsize, shuffle=False):
    while True:
        cachesize = index_queue.get()
        if cachesize is None:
            data_queue.put(None)
            break
        try:
            feats, targets = zip(*[(v, [labels[k]]*len(v)) for k, v in (next(datastream)
                                                               for _ in range(cachesize)) if k in labels])
        except StopIteration:
            # Just return the data
            pass
            # Returns Valueerror, if zip fails (list is empty, thus no data)
        except ValueError:
            data_queue.put(None)
            break
        assert feats is not None, "Check the labels!"
        # No features to return, just tell the iterator its finished
        feats = np.concatenate(feats)
        # Assuming multiple labels for each feature, targets has size 2xDATA ->
        # DATAx2
        targets = np.concatenate(targets)
        tnetdataset = TensorDataset(torch.from_numpy(feats),
                                    torch.from_numpy(targets).long())

        dataloader = DataLoader(
            tnetdataset, batch_size=batchsize,
            shuffle=shuffle, drop_last=shuffle)
        data_queue.put(dataloader)

class ASVLoader(object):
    """docstring for ASVLoader"""
    def __init__(self, stream, labels, countsfile, num_ouputs, cacheszie=200, batchsize=256, shiffle=False):
        super(ASVLoader, self).__init__()
        self.stream=stream
        self.labels=labels
        self.cachsize=cacheszie
        self.batchsize=batchsize
        self.shuffle=shiffle
        self.num_outputs=num_ouputs
        if isinstance(countsfile,str):
            with open(countsfile) as countsfileiter:
                self.lengths = {k:int(v) for k, v in (
                    l.rstrip('\n').split() for l in  countsfileiter)}
        else:
            self.lengths={k:int(v) for k,v in(
                l.rstrip('\n').split() for l in countsfile)}
        self.num_caches=int(
            max(np.ceil(1. * len(self.lengths) / self.cachsize), 1))
        self.nsamples = sum(self.lengths.values())

        key, feat = next(kaldi_io.read_mat_ark(self.stream))
        self.inputdim = feat.shape[-1]

    def __iter__(self):
        return itertools.chain.from_iterable(ASVLoaderIter(self))

    def __len__(self):
        return self.num_caches

class ASVLoaderIter(object):
    def __init__(self, loader):
        super(ASVLoaderIter,self).__init__()
        self.stream = loader.stream
        self.lengths = loader.lenghts
        self.cachesize = loader.cachesize
        self.labels = loader.labels
        self.shuffle = loader.shuffle
        self.batchsize = loader.batchsize
        self.nsamples = loader.nsamples
        self.num_caches = len(loader)
        self.idx = 0
        self.startWork()

    def _submitjob(self):
        self.idx+=1
        self.index_queue.put(self.cachesize)

    def startWork(self):
        self.data_queue = multiprocessing.SimpleQueue()
        self.index_queue = multiprocessing.SimpleQueue()
        self.worker = multiprocessing.Process(target=_fetch_caches, args=(kaldi_io.read_mat_ark(
            self.stream), self.index_queue, self.data_queue, self.labels, self.batchsize, self.shuffle))
        self.worker.start()
        self._submitjob()
    def _shutdown(self):
        self.index_queue.put(None)
        self.worker.join()
        self.worker.terminate()
        self.idx = -1

    def __del__(self):
        self._shutdown()

    def __len__(self):
        return self.nsamples

    def __next__(self):
        try:
            res = self.data_queue.get()
            if self.idx == -1 or not res:
                raise StopIteration
            if self.idx < self.num_caches:
                self._submitjob()
            elif self.idx >= self.num_caches:
                self._shutdown()
            return res

        except KeyboardInterrupt:
            self._shutdown()
            raise StopIteration
    next = __next__()

    def __iter__(self):
        return self
