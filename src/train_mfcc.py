import kaldi_io
import argparse
from tqdm import tqdm
import numpy as np
import torch
from tdataset import TensorDataset
from torch.utils.data import DataLoader
from CNN import cnn_fbank
import torchnet



def norm_arr(a):
    if a.shape[1] < 200:
        a = np.pad(a, ((0, 0), (0, 200 - a.shape[1])), 'edge')
    return a


def getDataloader(path, label, mode):
    feats, targets = zip(*[(norm_arr(mat.T[:, 0:200]), int(label[key]))
                           for key, mat in tqdm(kaldi_io.read_mat_scp(path))
                           if key in label])
    feats = np.array(feats)
    # feats = normalize(feats)
    targets = np.array(targets)
    assert len(feats) == len(targets)
    dataset = TensorDataset(torch.from_numpy(feats).float(),
                            torch.from_numpy(targets).long())
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    return dataloader, feats.shape[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("traindata", type=argparse.FileType("r"))
    parser.add_argument("trainlabel", type=argparse.FileType("r"))
    parser.add_argument("cvdata", type=argparse.FileType("r"))
    parser.add_argument("cvlabel", type=argparse.FileType("r"))
    args = parser.parse_args()
    trainlabel = {line.rstrip('\n').split()[0]: line.rstrip('\n').split()[1]
                  for line in args.trainlabel}

    cvlabel = {line.rstrip('\n').split()[0]: line.rstrip('\n').split()[1]
               for line in args.cvlabel}
    trainloader, inputdim = getDataloader(
        args.traindata, trainlabel, mode='train')
    cvloader, inputdim = getDataloader(args.cvdata, cvlabel, mode='dev')
    print (inputdim)

    for feats, label in tqdm(trainloader):
        print(feats.size())
        print(label.size())



if __name__ == "__main__":
    main()
