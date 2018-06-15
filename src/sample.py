import torch
import os

import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from vocabulary import Vocabulary
from model import EncoderCNN,DecoderRNN
from PIL import  Image

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def load_image(imgae_path, transform):
    image=Image.open(imgae_path)
    image=image.resize([224,224],Image.LANCZOS)

    if transform is not None:
        image=transform(image).unsqueeze(0)

    return image

def main(args):
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    vocab=Vocabulary.load_vocab(args['data_dir'])
    args['vocab_size']=len(vocab)
    encoder=EncoderCNN(args).eval()
    decoder=DecoderRNN(args)
    encoder.to(device)
    decoder.to(device)

    encoder.load_state_dict(torch.load(os.path.join(args['model_dir'],args['encoder_name'])))
    decoder.load_state_dict(torch.load(os.path.join(args['model_dir'],args['decoder_name'])))

    test_caption_list=[]
    for file_name in os.listdir(os.path.join(args['data_dir'],args['image_dir'])):
        if os.path.isfile(os.path.join(args['data_dir'],args['image_dir'],file_name)):
            image=load_image(os.path.join(args['data_dir'],args['image_dir'],file_name),transform)
            image_tensor=image.to(device)
        else:
            continue

        feature=encoder(image_tensor)
        sample_ids=decoder.sample(feature)
        sample_ids=sample_ids[0].cpu().numpy()

        sample_caption=[]
        for word_id in sample_ids:
            word=vocab.idx2word[word_id]
            sample_caption.append(word)
            if word == '<end>':
               break

        sentence=' '.join(sample_caption)
        print(sentence)
        test_caption_list.append((file_name,sentence))
#        image=Image.open(os.path.join(args['data_dir'],args['image_dir'],file_name))
#        plt.imshow(np.asarray(image))

    with open(os.path.join(args['data_dir'],'test_caption.txt'),'w') as f:
         for item in test_caption_list:
             f.write('image_name:{} ---- generated_caption:{}\n'.format(item[0],item[1]))
             f.write('\n')




if __name__=='__main__':

    opt={
        'data_dir': '../data_dir',
        'model_dir': '../data_dir/exp_2/checkpoints/2018_05_29_08_19_46',
        'caption_dir': '../raw_data_dir/annotations',
        'encoder_name':'encoder.pt',
        'decoder_name':'decoder.pt',
        'crop_size':224,
        'word_dime':256,
        'hidd_dime':512,
        'hidd_layer':1,
        'max_seq_leng':20,
        'image_dir':'../data_dir/test_image'
}

    main(opt)
