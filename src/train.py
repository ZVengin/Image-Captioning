import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pickle
import logging

from data_loader import get_loader
from model import EncoderCNN,DecoderRNN
from torch.nn.utils.rnn import  pack_padded_sequence
from torchvision import transforms
from tensorboard_logger import configure,log_value
from vocabulary import Vocabulary
from checkpoint import Checkpoint
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

#sys.path.insert(0,'/home/zvengin/Development/tool/YellowFin_Pytorch/tuner_utils')
#from yellowfin import YFOptimizer

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s -- %(message)s')
logger=logging.getLogger(__name__)


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):

    configure(os.path.join(args['exp_dir'],'log_dir'))

    transform =transforms.Compose([
        transforms.RandomCrop(args['crop_size']),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),
                             (0.229,0.224,0.225))])

    data_loader=get_loader({'data_dir' : args['data_dir'],
                             'exp_dir' : args['exp_dir'],
                             'raw_data_dir' : args['raw_data_dir'],
                             'batch_size' : args['batch_size'],
                             'transform' : transform,
                             'num_workers' : args['num_workers'],
                             'shuffle' : args['shuffle'],
                             'mode':'train'})

#    valid_data_loader=get_loader({'data_dir' : args['data_dir'],
#                             'raw_data_dir' : args['raw_data_dir'],
#                             'batch_size' : int(args['batch_size']/4),
#                             'transform' : transform,
#                             'num_workers' : args['num_workers'],
#                             'shuffle' : args['shuffle'],
#                             'mode':'validate'})

    args['vocab_size']=len(Vocabulary.load_vocab(args['exp_dir']))


    encoder=EncoderCNN(args).train()
    decoder=DecoderRNN(args).train()

    if args['pretrained']:
        checkpoint_path=Checkpoint.get_latest_checkpoint(args['exp_dir'])
        checkpoint=Checkpoint.load(checkpoint_path)
        encoder.load_state_dict(checkpoint.encoder)
        decoder.load_state_dict(checkpoint.decoder)
        step=checkpoint.step
        epoch=checkpoint.epoch
        omit=True

    else:
        step=0
        epoch=0
        omit=False

    encoder.to(device)
    decoder.to(device)

    criterion=nn.CrossEntropyLoss()
    params=list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
#    params=list(decoder.parameters()) + list(encoder.parameters())
    optimizer=torch.optim.Adam(params, lr=args['lr'])
    scheduler = StepLR(optimizer, step_size=40, gamma=0.1)
#    optimizer=YFOptimizer(params)

    total_step= len(data_loader)
    min_valid_loss=float('inf')

    for epoch in range(epoch,args['num_epochs']):
        scheduler.step()
        for idx, (images,captions,leng) in enumerate(data_loader):

            if omit:
                if idx<(step-total_step*epoch):
                   logger.info('idx:{},step:{}, epoch:{}, total_step:{}, diss:{}'.format(idx,step,epoch,total_step,step-total_step*epoch))
                   continue
                else: omit=False

            images = images.to(device)
            captions = captions.to(device)
            targets =pack_padded_sequence(captions,leng,batch_first=True)[0]

            features=encoder(images)
            outputs=decoder(features,captions, leng)
            loss=criterion(outputs,targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(),5)
            optimizer.step()

            log_value('loss',loss.item(),step)
            step+=1

            if step % args['log_step'] == 0:
                logger.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args['num_epochs'], idx, total_step, loss.item(), np.exp(loss.item())))

            if step % args['valid_step'] == 0:
#                valid_loss=validate(encoder.eval(),decoder,criterion,valid_data_loader)
#                if valid_loss<min_valid_loss:
#                    min_valid_loss=valid_loss
                Checkpoint(encoder,decoder,optimizer,epoch,step).save(args['exp_dir'])
#                logger.info('Epoch [{}/{}], step [{}/{}], valid_loss: {:.4f}, perplexity: {:5.4f}'.format(
#                    epoch,args['num_epochs'],step,total_step,valid_loss,np.exp(valid_loss)
#                ))
#                log_value('valid_loss',valid_loss,int(step/args['valid_step']))

def validate(encoder,decoder,criterion,data_loader):
    valid_loss=0

    for idx,(images,captions,leng) in enumerate(data_loader):
        images=images.to(device)
        captions=captions.to(device)
        targets=pack_padded_sequence(captions,leng,batch_first=True)[0]

        features=encoder(images)
        outputs=decoder(features,captions,leng)
        valid_loss+=criterion(outputs,targets).cpu().item()

    valid_loss/=len(data_loader)

    return valid_loss



