import torch
import numpy as np
import argparse
import kaldi_io
from tqdm import tqdm
from tdataset import TensorDataset
import torchnet as tnt
from DNN import DNN
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.preprocessing import normalize
from librosaWav import get_feats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('traindata', type=lambda  x: {k:v for k,v in kaldi_io.read_mat_ark(x)})
    parser.add_argument('trainlabel',type=str)
    parser.add_argument('cvdata', type=lambda  x:{k:v for k,v in kaldi_io.read_mat_ark(x)})
    parser.add_argument('cvlabel',type=str)
    parser.add_argument('--nocuda', action="store_true", default=False)

    args = parser.parse_args()


    with open(args.trainlabel) as flabel:
        trainlabel = {line.rstrip('\n').split()[0] : line.rstrip('\n').split()[1] for line in flabel}

    with open(args.cvlabel) as flabel:
        cvlabel = {line.rstrip('\n').split()[0] : line.rstrip('\n').split()[1] for line in flabel}

    feats, targets =zip(*[(v, [int(trainlabel[k])] * len(v)) for k, v in args.traindata.items()])
    #feats, targets = get_feats('/home/slhome/jqg01/work-home/workspace/asvspoof2017/ASVspoof2017_train/', trainlabel)
    feats = np.concatenate(feats)
    targets = np.concatenate(targets)

    feats = normalize(feats, axis=0, norm='l2')

    assert len(feats) == len(targets)
    tnetdataset = TensorDataset(torch.from_numpy(feats).float(),
                                torch.from_numpy(targets).long())

    trainloader = DataLoader(tnetdataset, batch_size=1024, shuffle=True)




    cvloader = [(v, int(cvlabel[k])) for k, v in args.cvdata.items()]
    '''
    feats, targets =zip(*[(v, [int(cvlabel[k])] * len(v)) for k, v in args.cvdata.items()])
    feats, targets = get_feats('/home/slhome/jqg01/work-home/workspace/asvspoof2017/ASVspoof2017_dev/', cvlabel)
    feats = np.concatenate(feats)
    targets = np.concatenate(targets)
    tnetdataset = TensorDataset(torch.from_numpy(feats).float(),
                                torch.from_numpy(targets).long())
    cvloader = DataLoader(tnetdataset, batch_size=1024, shuffle=False)
    '''
    assert len(feats) == len(targets)


    dnn = DNN(feats.shape[1],2)

    if not args.nocuda:
        dnn.cuda()

    def lossfun(sample):
        x,y = sample
        x,y = torch.autograd.Variable(x), torch.autograd.Variable(y)
        if not args.nocuda:
            x, y = x.cuda(), y.cuda()

        outputs = dnn(x)
        loss = criterion(outputs,y)
        return loss, outputs

    def evalfun(sample):
        rx,ry = sample
        x = torch.from_numpy(rx).float()
        y = torch.from_numpy(np.array([ry]*len(rx))).long()

        if not args.nocuda:
            x,y= x.cuda(), y.cuda()
        x_var, y_var = torch.autograd.Variable(x, volatile=True), torch.autograd.Variable(y, volatile=True)
        outputs = dnn(x_var)
        loss = criterion(outputs, y_var)
        res = outputs.cpu().data.numpy()
        predicted = 1 if len(res[res[:,1]>=res[:,0]]) >= 0.5 * len(res) else 0

        acc = 1 if predicted == ry else 0

        return {'acc': acc, 'loss':loss}, outputs

    def on_start_epoch(state):
        dnn.train()
        meter_loss.reset()
        state['iterator'] = tqdm(state['iterator'])

    def on_forward(state):
        meter_loss.add(state['loss'].data[0])

    def on_forward_eval(state):
        meter_loss.add(state['loss']['acc'])
        meter_loss2.add(state['loss']['loss'].data[0])

    def on_end_epoch(state):
        trainmessage='Train Epoch:{:>3d}: Time:{:=6.1f}s/{:=4.1f}m Loss: {:=.4f} LR: {:=3.1e}'.format(
            state['epoch'],time_meter.value(), time_meter.value()/60,meter_loss.value()[0],optimizer.param_groups[0]['lr'])
        print(trainmessage)
        meter_loss.reset()
        meter_loss2.reset()
        dnn.eval()
        engine.hooks['on_forward']= on_forward_eval
        engine.test(evalfun,tqdm(cvloader))
        engine.hooks['on_forward'] = on_forward
        acc, loss = meter_loss.value()[0], meter_loss2.value()[0]
        #loss = meter_loss.value()[0]
        evalmessage='CV Epoch {:>3d}: acc:{:=.4f} loss:{:=.4f}'.format(state['epoch'],acc,loss)
        #evalmessage = 'CV Epoch {:>3d}: loss:{:=.4f}'.format(state['epoch'], loss)
        print (evalmessage)
        sched.step(acc)

    engine = tnt.engine.Engine()
    criterion = torch.nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(dnn.parameters(),lr=1e-3,nesterov=True,momentum=0.9)
    optimizer = torch.optim.Adam(params=dnn.parameters(),lr=1e-3)
    sched = ReduceLROnPlateau(optimizer,mode="max",factor=0.5,patience=2)
    time_meter = tnt.meter.TimeMeter(False)
    meter_loss = tnt.meter.AverageValueMeter()
    meter_loss2 = tnt.meter.AverageValueMeter()

    engine.hooks['on_start_epoch']=on_start_epoch
    engine.hooks['on_end_epoch']=on_end_epoch
    engine.hooks['on_forward'] = on_forward
    engine.train(lossfun,trainloader,maxepoch=100,optimizer=optimizer)

if __name__ == '__main__':
    main()
