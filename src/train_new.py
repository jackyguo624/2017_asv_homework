import torch
import numpy as np
import argparse
import kaldi_io
from tqdm import tqdm
from tdataset import TensorDataset
import torchnet as tnt
import CNN
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torch.utils.data import DataLoader
from sklearn.preprocessing import normalize
from librosaWav import get_feats
import msvgg
from dataset import ASVLoader
from os import listdir
from os.path import isfile, join
import scipy.io as sio
import os
import ResNet
import VGG

def normalize(x):
    return (x) / np.linalg.norm(x)

def extend_frame(feats, n_ext=5):
    nfeats = []
    for feature in feats:
        feature = np.pad(feature, [[n_ext, n_ext], [0, 0]], mode="edge")
        records = []
        for i in range(n_ext, feature.shape[0]-n_ext):
            tmp_feature = feature[i-n_ext:i+n_ext+1, :].reshape(-1)
            records.append(tmp_feature)
        nfeats.append(np.array(records))
    return nfeats

def save_checkpoint(state,is_best, savedir):
    filename = os.path.join(savedir, 'checkpoint.th')
    # torch.save(state, filename)
    if is_best:
        torch.save(state['model'].state_dict(),os.path.join(savedir,'model_best.param'))
        print('save the best model at {0} at epoch {1} with acc {2}'
              .format(savedir, state['epoch'], state['acc']))
        acc_name = os.path.join(savedir, 'acc')
        with open(acc_name, 'w') as f:
            f.write('acc: {}, loss{}  at epoch: {}'.
                    format(state['acc'], state['loss'], state['epoch']))


def norm_arr(a):
    if a.shape[1] < 200:
        a = np.pad(a, ((0, 0), (0, 200 - a.shape[1])), 'edge')
    return a


def getDataloader(path, label, model, mode):
    feats, targets = (None, None)
    print (model)
    if 'cqcc' in model:
        feats, targets = zip(*[(norm_arr(sio.loadmat(join(path, f))['CQcc'][0:20, 0:200]),
                                int(label[f])) for f in tqdm(listdir(path)) if f in label])

    elif 'mfcc' in model or 'fbank' in model or 'fft' in model:
        feats, targets = zip(*[(norm_arr(mat.T[:, 0:200]), int(label[key]))
                               for key, mat in tqdm(kaldi_io.read_mat_scp(path))
                               if key in label])

    assert feats is not None
    assert targets is not None
    feats = np.array(feats)
    # feats = normalize(feats)
    targets = np.array(targets)
    assert len(feats) == len(targets)
    dataset = TensorDataset(torch.from_numpy(feats).float(),
                                torch.from_numpy(targets).long())
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return dataloader, feats.shape[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('traindata', type=str)
    parser.add_argument('trainlabel', type=argparse.FileType('r'))
    parser.add_argument('cvdata', type=str)
    parser.add_argument('cvlabel', type=argparse.FileType('r'))
    parser.add_argument('-model', type=str, default='cnn')
    # parser.add_argument('trcount', type=argparse.FileType('r'))
    # parser.add_argument('cvcount', type=argparse.FileType('r'))
    parser.add_argument('--nocuda', action="store_true", default=False)
    parser.add_argument('-o', '--output',type=str,help="Output directory, If none its output/MODELNAME")

    args = parser.parse_args()
    args.output = args.output if args.output else "output/"+args.model

    trainlabel = {line.rstrip('\n').split()[0]: line.rstrip('\n').split()[1]
                  for line in args.trainlabel}

    cvlabel = {line.rstrip('\n').split()[0]: line.rstrip('\n').split()[1]
               for line in args.cvlabel}

    trainloader, inputdim = getDataloader(args.traindata,
                                          trainlabel, model=args.model,
                                          mode='train')
    cvloader, inputdim = getDataloader(args.cvdata, cvlabel,
                                       model=args.model, mode='dev')
    if 'cnn' in args.model:
        if 'fft' in args.model:
            model = getattr(VGG, args.model)(inputdim, 2)
        else:
            model = getattr(CNN, args.model)(inputdim, 2)
    elif 'res' in args.model:
        model = getattr(ResNet, args.model)()

    '''
    best_model_path = os.path.join(args.output, "model_best.param")
    if os.path.isfile(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print ("Load model at {}".format(best_model_path))
    '''
    if not args.nocuda:
        model.cuda()

    def lossfun(sample):
        x, y = sample
        x, y = torch.autograd.Variable(x), torch.autograd.Variable(y)
        if not args.nocuda:
            x, y = x.cuda(), y.cuda()
        outputs = model(x)
        loss = criterion(outputs, y)
        return loss, outputs

    def evalfun(sample):
        x, y = sample
        # x = torch.from_numpy(rx).float()
        # y = torch.from_numpy(np.array([ry]*len(rx))).long()
        if not args.nocuda:
            x, y = x.cuda(), y.cuda()
        x_var, y_var = torch.autograd.Variable(x, volatile=True), torch.autograd.Variable(y, volatile=True)
        outputs = model(x_var)
        loss = criterion(outputs, y_var)
        # res = outputs.cpu().data.numpy()
        # predicted = 1 if len(res[res[:, 1] >= res[:, 0]]) >= 0.5 * len(res) else 0
        _, predicted = torch.max(outputs.data, 1)
        total = y.size(0)
        correct = (predicted == y).sum()
        res1 = (len(torch.nonzero(predicted))) * 1.0
        print res1, total, res1 * 1. / total
        res1 = (len(torch.nonzero(predicted))) * 1.0 / total
        acc = correct * 1. / total
        return {'acc': acc, 'loss': loss, 'res1': res1}, outputs
    
    def on_start_epoch(state):
        model.train()
        meter_loss.reset()
        meter_loss2.reset()
        state['iterator'] = tqdm(state['iterator'])

    def on_forward(state):
        meter_loss.add(state['loss'].data[0])

    def on_forward_eval(state):
        meter_loss.add(state['loss']['acc'])
        meter_loss2.add(state['loss']['loss'].data[0])
        res1.add(state['loss']['res1'])

    def on_start(state):
        state['best_acc'] = state[
            'best_acc'] if 'best_acc' in state else 0.5
        state['best_loss'] = state[
            'best_loss'] if 'best_loss' in state else 1e10

    def on_end_epoch(state):
        trainmessage='Train Epoch:{:>3d}: Time:{:=6.1f}s/{:=4.1f}m Loss: {:=.4f} LR: {:=3.1e}'.format(
            state['epoch'],time_meter.value(), time_meter.value()/60,meter_loss.value()[0],optimizer.param_groups[0]['lr'])
        print(trainmessage)
        meter_loss.reset()
        meter_loss2.reset()
        res1.reset()
        model.eval()
        engine.hooks['on_forward']= on_forward_eval
        engine.test(evalfun,tqdm(cvloader))
        engine.hooks['on_forward'] = on_forward
        acc, loss = meter_loss.value()[0], meter_loss2.value()[0]

        evalmessage='CV Epoch {:>3d}: acc:{:=.4f} loss:{:=.4f} res1: {:=.8f}'.format(
            state['epoch'], acc, loss, res1.value()[0])

        print (evalmessage)
        
        isbest = acc > state['best_acc']
        state['best_acc'] = acc if isbest else state['best_acc']
        '''isbest = loss < state['best_loss']
        state['best_loss'] = loss if isbest else state['best_loss']'''

        isbest = True
        save_checkpoint({
            'model': model,
            'epoch': state['epoch'],
            'acc': acc,
            'loss': loss
        }, isbest, args.output
        )

        # sched.step(acc)
        sched.step()
    engine = tnt.engine.Engine()
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(),lr=1e-2,nesterov=True, weight_decay=1e-6, momentum=0.9)
    optimizer = torch.optim.Adam(params=model.parameters(),lr=0.01)
    # sched = ReduceLROnPlateau(optimizer,mode="max",factor=0.5,patience=2)
    sched = MultiStepLR(optimizer, milestones=[5, 15, 25, 35, 45], gamma=0.5)
    #sched = MultiStepLR(optimizer, milestones=[3, 7], gamma=0.1)
    time_meter = tnt.meter.TimeMeter(False)
    meter_loss = tnt.meter.AverageValueMeter()
    meter_loss2 = tnt.meter.AverageValueMeter()
    res1 = tnt.meter.AverageValueMeter()
    engine.hooks['on_start'] = on_start
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.hooks['on_forward'] = on_forward
    engine.train(lossfun, trainloader, maxepoch=50, optimizer=optimizer)


if __name__ == '__main__':
    main()


'''
   feats, targets =zip(*[(v, [int(trainlabel[k])] * len(v)) for k, v in args.traindata.items()])
    #feats, targets = get_feats('/home/slhome/jqg01/work-home/workspace/asvspoof2017/ASVspoof2017_train/', trainlabel)
    feats = np.concatenate(feats)
    targets = np.concatenate(targets)

    feats = normalize(feats, axis=0, norm='l2')

    assert len(feats) == len(targets)
    tnetdataset = TensorDataset(torch.from_numpy(feats).float(),
                                torch.from_numpy(targets).long())

    trainloader = DataLoader(tnetdataset, batch_size=1024, shuffle=True)

o    cvloader = [(v, int(cvlabel[k])) for k, v in args.cvdata.items()]

    assert len(feats) == len(targets)

    trainloader = ASVLoader(args.traindata,trainlabel,args.trcount,2, mode='train',shiffle=True)
    cvloader = ASVLoader(args.cvdata, cvlabel, args.cvcount, 2, mode='dev',
                         shiffle=False)


'''
