import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from os.path import join
from os import listdir
from tdataset import TensorDataset
import argparse
import CNN
import scipy.io as sio
import os
import sys
import errno


def normalize(x):
    return (x) / np.linalg.norm(x)


def norm_arr(a):
    if a.shape[1] < 200:
        a = np.pad(a, ((0, 0), (0, 200 - a.shape[1])), 'edge')
    return a


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class IDforamter(object):
    def __init__(self, mode):
        self.mode = mode

    def hash(self, idx):
        return int(idx[2:-4])

    def rehash(self, hashcode):
        prefix = 'T_' if self.mode is 'train' else 'D_' if self.mode is 'dev' else 'E_'
        return prefix+str(hashcode)+'.wav'


def getDataloader(path, label, mode):

    idform = IDforamter(mode)
    ids, feats = zip(*[(idform.hash(f), norm_arr(sio.loadmat(join(path, f))[
        'CQcc'][0:20, 0:200]))
        for f in tqdm(listdir(path)) if f in label])
    feats = np.array(feats)
    # feats = normalize(feats)
    ids = np.array(ids)
    assert len(ids) == len(feats)
    dataset = TensorDataset(torch.from_numpy(ids),
                            torch.from_numpy(feats).float())
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    return dataloader, feats.shape[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('evaldata', type=str)
    parser.add_argument('evallabel', type=argparse.FileType('r'))
    parser.add_argument('-model', type=str, default='cnn')
    parser.add_argument('--nocuda', action="store_true", default=False)
    parser.add_argument('-o', '--output', type=str)
    parser.add_argument('-m', '--mode', type=str)

    args = parser.parse_args()

    args.output = args.output if args.output else "predict/"+args.model

    if sys.version_info[0] <= 3.2:
        mkdir_p(args.output)
    else:
        os.makedirs(args.output, exist_ok=True)

    evallabel = {line.rstrip('\n').split()[0]:
                 line.rstrip('\n').split()[1] for line in args.evallabel}
    evalloader, inputdim = getDataloader(args.evaldata, evallabel, mode=args.mode)
    model = getattr(CNN, args.model)(inputdim, 2)
    model.load_state_dict(torch.load(os.path.join(
        "output", args.model, 'model_best.param')))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    if not args.nocuda:
        model.cuda()

    # idform = IDforamter('dev')
    outputfile = open(args.output+"/score.dat", 'ab')
    softmax = torch.nn.Softmax()
    for k, x in tqdm(evalloader):
        x = torch.autograd.Variable(x, volatile=True)
        if not args.nocuda:
            x = x.cuda()
        outputs = model(x)
        outputs = softmax(outputs)
        _, predicted = torch.max(outputs.data, 1)
        total = k.size(0)
        res1 = (len(torch.nonzero(predicted))) * 1.0
        print res1, total, res1 * 1. / total
        scores = outputs.data.cpu().numpy()
        keys = k.numpy()
        # keys = np.array([idform.rehash(k) for k in keys])
        keys = np.array([keys])
        res = np.concatenate([keys, scores.T])
        np.savetxt(outputfile, res.T, fmt='%d %.8f %.8f')
    outputfile.close()


if __name__ == "__main__":
    main()
