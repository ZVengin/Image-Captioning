import nltk
import json
import os
import torch

from model import EncoderCNN,DecoderRNN
from torchvision import transforms
from data_loader import Dataset
from nltk.translate import bleu_score
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

# args={
#       'data_dir',
#       'caption_dir',
#       'num_workers',
#       'shuffle',
#       'mode',
#       'word_dime',
#       'hidd_dime',
#       'hidd_layer_num',
#       'vocab_size',
#       'max_seq_len',
#       'model_path'}
#
#device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(args):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    dataset = Dataset({'data_dir' : args['data_dir'],
                       'exp_dir':args['exp_dir'],
                       'raw_data_dir' : args['raw_data_dir'],
                       'transform' : transform,
                       'mode':'test'})
    args['vocab_size']=len(dataset.vocab)

    encoder = EncoderCNN(args).eval()
    decoder = DecoderRNN(args).eval()

    encoder=encoder.to(device)
    decoder=decoder.to(device)

    encoder.load_state_dict(torch.load(os.path.join(args['model_path'],'encoder.pt')))
    decoder.load_state_dict(torch.load(os.path.join(args['model_path'],'decoder.pt')))

    generated_captions=[]
    image_ids=[]
    target_captions=[]

    for idx in range(len(dataset.ids)):
        image_id,image,captions=dataset.get_test_item(idx)
        image=image.to(device)
        print(idx)

        features=encoder(image)
        word_ids=decoder.sample(features)
        word_ids=word_ids[0].cpu().tolist()

        words=[]
        for word_id in word_ids:
            if dataset.vocab.idx2word[word_id] == '<start>':
               continue
            if dataset.vocab.idx2word[word_id] != '<end>':
                words.append(dataset.vocab.idx2word[word_id])
            else:
                break
        image_ids.append(image_id)
        generated_captions.append(words)
        target_captions.append(captions)
        print(words)

    BLEU_score=bleu_score.corpus_bleu(target_captions,generated_captions,weights=(0.25,0.25,0.25,0.25))

    image_captions = [{'image_id': image_ids[idx], 'caption': ' '.join(generated_captions[idx])} for idx in
                      range(len(image_ids))]

    captions_path=os.path.join(args['exp_dir'],'generated_captions.txt')
    bleu_score_path=os.path.join(args['exp_dir'],'bleu_score.txt')
    image_caption_path=os.path.join(args['exp_dir'],'image_captions.json')

    with open(captions_path,'w') as f:
        for idx in range(len(generated_captions)):
            f.write('*'*50+'\n')
            f.write('-'*20+'generated_captions'+'-'*20+'\n')
            f.write(' '.join(generated_captions[idx])+'\n')
            f.write('-'*20+'target_captions'+'-'*20+'\n')
            for words in target_captions[idx]:
                f.write(' '.join(words)+'\n')
            f.write('*'*50+'\n')
            f.write('\n')

    with open(bleu_score_path,'w') as f:
        f.write('BLEU_score: {}'.format(str(BLEU_score)))

    with open(image_caption_path,'w') as f:
        json.dump(image_captions,f)

#    compute_score(os.path.join(args['raw_data_dir'],'annotations','captions_val2014.json'),image_caption_path,
#                  os.path.join(args['data_dir'],'evaluation_score.txt'))


def evaluate_with_beam_search(args):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    dataset = Dataset({'data_dir' : args['data_dir'],
                       'exp_dir':args['exp_dir'],
                       'raw_data_dir' : args['raw_data_dir'],
                       'transform' : transform,
                       'mode':'test'})
    args['vocab_size']=len(dataset.vocab)

    encoder = EncoderCNN(args).eval()
    decoder = DecoderRNN(args).eval()

    encoder=encoder.to(device)
    decoder=decoder.to(device)

    encoder.load_state_dict(torch.load(os.path.join(args['model_path'],'encoder.pt')))
    decoder.load_state_dict(torch.load(os.path.join(args['model_path'],'decoder.pt')))

    generated_captions=[]
    image_ids=[]
    target_captions=[]

    for idx in range(len(dataset.ids)):
        image_id,image,captions=dataset.get_test_item(idx)
        image=image.to(device)
        print(idx)

        features=encoder(image)
        generated_sents=decoder.decode_with_beam_search(features)
#        print(generated_sents)
        sents=[]
        for sent_id in generated_sents:
            words=[]
            for word_id in sent_id[0]:
                if dataset.vocab.idx2word[word_id]=='<start>':
                   continue
                elif dataset.vocab.idx2word[word_id]!='<end>':
                   words.append(dataset.vocab.idx2word[word_id])
                else:
                   break

            sents.append((' '.join(words),sent_id[1]/len(sent_id[0])))
        sents=sorted(sents,key=lambda x:x[1],reverse=True)
        generated_captions.append(sents)
        image_ids.append(image_id)
        target_captions.append(captions)

    image_captions=[{'image_id':image_ids[idx],'caption':generated_captions[idx][0][0]} for idx in range(len(image_ids))]    

    captions_path=os.path.join(args['exp_dir'],args['model_dir'],'generated_captions_beam_3.txt')
#    bleu_score_path=os.path.join(args['exp_dir'],'bleu_score.txt')
    image_caption_path=os.path.join(args['exp_dir'],args['model_dir'],'image_captions_beam_3.json')

    with open(captions_path,'w') as f:
        for idx in range(len(generated_captions)):
            f.write('*'*50+'\n')
            f.write('-'*20+'generated_captions'+'-'*20+'\n')
            for sent in generated_captions[idx]:
                f.write(sent[0]+'\n')
            f.write('-'*20+'target_captions'+'-'*20+'\n')
            for words in target_captions[idx]:
                f.write(' '.join(words)+'\n')
            f.write('*'*50+'\n')
            f.write('\n')

    with open(image_caption_path,'w') as f:
        json.dump(image_captions,f)
   
            
   

def compute_score(annotation_file,fake_caption_file,score_file):
    coco=COCO(annotation_file)
    cocoRes=coco.loadRes(fake_caption_file)
    cocoEval=COCOEvalCap(coco,cocoRes)
    
    cocoEval.params['image_id']=cocoRes.getImgIds()

    cocoEval.evaluate()

    with open(score_file,'w') as f:
        for metric,score in cocoEval.eval.items():
            f.write('Metric: {}, Score: {}\n'.format(metric,str(score)))


model_dir='2018_06_11_10_18_48'
exp_dir='../data_dir/exp_12'

args={
      'data_dir':'../data_dir',
      'exp_dir':exp_dir,
      'raw_data_dir':'../raw_data_dir/annotations',
      'num_workers':1,
      'shuffle':True,
      'mode':'test',
      'word_dime':512,
      'hidd_dime':512,
      'hidd_layer':1,
      'max_seq_leng':20,
      'model_dir':model_dir,
      'model_path':os.path.join(exp_dir,'checkpoints',model_dir)}
#evaluate_with_beam_search(args)
#evaluate(args)
compute_score(os.path.join(args['raw_data_dir'],'captions_val2014.json'),
              os.path.join(args['exp_dir'],args['model_dir'],'image_captions_beam_3.json'),
              os.path.join(args['exp_dir'],args['model_dir'],'evaluation_score_beam_search_3.txt'))
