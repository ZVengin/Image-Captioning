import torch
import torch.nn as nn
import torchvision.models as models

from torch.nn.utils.rnn import pack_padded_sequence

class EncoderCNN(nn.Module):
    def __init__(self,args):
        super(EncoderCNN,self).__init__()
        resnet=models.resnet152(pretrained=True)
        modules=list(resnet.children())[:-1]
        self.resnet=nn.Sequential(*modules)
        self.linear=nn.Linear(resnet.fc.in_features,args['word_dime'])
        self.bn=nn.BatchNorm1d(args['word_dime'],momentum=0.01)

    def forward(self,images):
        with torch.no_grad():
            features=self.resnet(images)
        features=features.reshape(features.size(0),-1)
        features=self.bn(self.linear(features))
        return features

class DecoderRNN(nn.Module):
    def __init__(self,args):
        super(DecoderRNN,self).__init__()
        self.word_dime=args['word_dime']
        self.hidd_dime=args['hidd_dime']
        self.hidd_layer=args['hidd_layer']
        self.vocab_size=args['vocab_size']

        self.embed=nn.Embedding(self.vocab_size,self.word_dime)
        self.rnn=nn.LSTM(self.word_dime,self.hidd_dime,self.hidd_layer,batch_first=True)
        self.linear=nn.Linear(self.hidd_dime,self.vocab_size)
        self.max_seq_leng=args['max_seq_leng']

    def forward(self, features,caption,leng):
        embed_captn=self.embed(caption)
        embed_captn=torch.cat((features.unsqueeze(1),embed_captn),1)
        pack_captn=torch.nn.utils.rnn.pack_padded_sequence(embed_captn,leng,batch_first=True)
        hidd_state,_=self.rnn(pack_captn)
        outputs=self.linear(hidd_state[0])
        return outputs

    def sample(self,features,hidd_state=None):
        sample_ids=[]
        words=features.unsqueeze(1)

        for idx in range(self.max_seq_leng):
            outs, hidd_state=self.rnn(words,hidd_state)
            outs=self.linear(outs.squeeze(1))
            _,pred_words=outs.max(1)
            sample_ids.append(pred_words)
            words=self.embed(pred_words)
            words=words.unsqueeze(1)
        sample_ids=torch.stack(sample_ids,1)

        return sample_ids
'''
encoder=EncoderCNN({'embed_size':128})
decoder=DecoderRNN({'word_dime':128,'hidd_dime':256,'hidd_layer':1,'vocab_size':50000,'max_seq_leng':20})
encoder.cuda()
decoder.cuda()
print(encoder)
print(decoder)
'''
