import torch
import torch.nn as nn
import torchvision.models as models

from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable

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

        self.drop=nn.Dropout(p=0.2)
        self.embed=nn.Embedding(self.vocab_size,self.word_dime)
        self.rnn=nn.LSTM(self.word_dime,self.hidd_dime,self.hidd_layer,batch_first=True)
        self.linear=nn.Linear(self.hidd_dime,self.vocab_size)
        self.max_seq_leng=args['max_seq_leng']

    def forward(self, features,caption,leng):
        embed_captn=self.drop(self.embed(caption))
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

    def decode_with_beam_search(self,features,hidden_vector=None):
        max_len=25
        beam_size=3
        dead_num=0
        eos_idx=2
        features=features.unsqueeze(1)

        generated_sents=[]
        
        inp=features.repeat(beam_size,1,1)  
        
        if torch.cuda.is_available(): 
            inp=inp.cuda()
            
            
        for step in range(max_len):
            
            out_step,hidden_vector=self.rnn(inp,hidden_vector)
            hidd_fact=hidden_vector[0].size(0)
            out_step=nn.functional.log_softmax(self.linear(out_step.squeeze(1)))
            word_prob=out_step.topk(beam_size)[0]
            word_idx=out_step.topk(beam_size)[1]
            
            tmp_words=[]
            
            if step==0:
                for idx in range(beam_size):
                    word=[]
                    word.append([word_idx.data[0][idx].tolist()])
                    word.append([word_prob.data[0][idx].tolist()])
                    word.append([hidden_vector[0].transpose(0,1).data[idx],
                                 hidden_vector[1].transpose(0,1).data[idx]])
                    
                    tmp_words.append(word)
                
                
            else:
                for idx in range(beam_size-dead_num):
                    for sub_idx in range(beam_size):
                        word=[]
                        word.append(candidate_words[idx][0]+[word_idx.data[idx][sub_idx].tolist()])
                        word.append([candidate_words[idx][1][0]+
                                     word_prob.data[idx][sub_idx].tolist()])
                        word.append([hidden_vector[0].transpose(0,1).data[idx],
                                     hidden_vector[1].transpose(0,1).data[idx]])
                        
                        tmp_words.append(word)

                        
            #print(step)

            tmp_words=sorted(tmp_words,key=lambda x: x[1][0]/len(x[0]), reverse=True)

            tmp_words=tmp_words[0:beam_size-dead_num]

            #if step==5:
                #print(tmp_words)

            candidate_words=[]

            #print(tmp_words)

            for idx in range(len(tmp_words)):
                if tmp_words[idx][0][-1]==eos_idx:
                    generated_sents.append((tmp_words[idx][0],tmp_words[idx][1][0]))
                    dead_num+=1
                    #print('generted words')
            
                else:
                    candidate_words.append(tmp_words[idx])
                    
            #print(candidate_words)
            
            if beam_size-dead_num==0: 
                return generated_sents


            hidden_vector=(Variable(torch.zeros(beam_size-dead_num,hidd_fact,hidden_vector[0].size(2))),
                           Variable(torch.zeros(beam_size-dead_num,hidd_fact,hidden_vector[0].size(2))))
            inp=Variable(torch.LongTensor([0]*(beam_size-dead_num)))


            for idx in range(beam_size-dead_num):
                hidden_vector[0].data[idx]=candidate_words[idx][2][0]
                hidden_vector[1].data[idx]=candidate_words[idx][2][1]
                inp.data[idx]=candidate_words[idx][0][-1]

            hidden_vector=(hidden_vector[0].transpose(0,1).contiguous(),
                           hidden_vector[1].transpose(0,1).contiguous())

            if torch.cuda.is_available():
                hidden_vector=(hidden_vector[0].cuda(),hidden_vector[1].cuda())
                inp=inp.cuda()

            inp=inp.contiguous().view(beam_size-dead_num,1)
            inp=self.embed(inp)


            #if step==1:
            #print(candidate_words)
            #print(inp)

        if step==max_len-1:
            for idx in range(len(candidate_words)):
                generated_sents.append((candidate_words[idx][0],candidate_words[idx][1][0]))
                
#        print(generated_sents)

        return generated_sents
'''
encoder=EncoderCNN({'embed_size':128})
decoder=DecoderRNN({'word_dime':128,'hidd_dime':256,'hidd_layer':1,'vocab_size':50000,'max_seq_leng':20})
encoder.cuda()
decoder.cuda()
print(encoder)
print(decoder)
'''
