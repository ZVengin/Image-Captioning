import torch
import os
import pickle
import nltk
import json
import logging

from PIL import Image
from pycocotools.coco import COCO
from vocabulary import Vocabulary

import torch.utils.data as data

logging.basicConfig(level=logging.DEBUG,format='%(message)s')
logger=logging.getLogger(__name__)

class Dataset(data.Dataset):

    def __init__(self,args):
        self.data_dir=args['data_dir']
        self.mode=args['mode']

#        self.vocab=Vocabulary.load_vocab(args['exp_dir'])
        with open(os.path.join(args['exp_dir'],'vocabulary.pkl'),'rb') as f:
             self.vocab = pickle.load(f)
        self.transform = args['transform']

        if args['mode']=='train':
            coco_train = COCO(os.path.join(args['raw_data_dir'], 'captions_train2014.json'))
            with open(os.path.join(args['exp_dir'],'train_annotations.json')) as f:
                 self.ids=json.load(f)
#        elif args['mode'] == 'validate':
            coco_eval = COCO(os.path.join(args['raw_data_dir'], 'captions_val2014.json'))
#            with open(os.path.join(self.data_dir,'validate_annotations.json')) as f:
#                 self.ids=json.load(f)
#                  self.ids+=json.load(f)
            self.coco=COCO()
            self.coco.anns.update(coco_train.anns)
            self.coco.anns.update(coco_eval.anns)
            self.coco.imgs.update(coco_train.imgs)
            self.coco.imgs.update(coco_eval.imgs)

        elif args['mode'] == 'test':
            self.coco = COCO(os.path.join(args['raw_data_dir'], 'captions_val2014.json'))
            with open(os.path.join(args['exp_dir'],'test_images.json')) as f:
                 self.ids=json.load(f)

        else:
            logger.info('illegal mode of loading dataset:{}'.format(args['mode']))
            return -1

    def __getitem__(self, index):
        coco = self.coco
        vocab = self.vocab
        ann_id = int(self.ids[index])
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']

        path = coco.loadImgs(img_id)[0]['file_name']

#        if self.mode == 'train':
        if os.path.exists(os.path.join(self.data_dir,'train_images', path)):
            image = Image.open(os.path.join(self.data_dir,'train_images', path)).convert('RGB')
#        elif self.mode == 'validate':
        elif os.path.exists(os.path.join(self.data_dir,'validate_images', path)):
            image = Image.open(os.path.join(self.data_dir,'validate_images',path)).convert('RGB')
        else:
            logger.info('image_id: {} doesn\'t exists!!!'.format(path))
            return -1

        if self.transform is not None:
            image = self.transform(image)

        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption=[]
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)

        return image,target

    def __len__(self):
        return len(self.ids)

    def get_test_item(self,index):
        img_id = self.ids[index]
        ann_list = [nltk.tokenize.word_tokenize(str(ann['caption']).lower())
                    for ann in self.coco.imgToAnns[img_id]]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.data_dir,'validate_images',path)).convert('RGB')
        img = img.resize([224,224],Image.LANCZOS)

        if self.transform is not None:
            img = self.transform(img).unsqueeze(0)

        return img_id,img,ann_list





def collate_fn(data):

# This function aims to construct a mini-batch

    data.sort(key=lambda x: len(x[1]), reverse=True)
    images,captions = zip(*data)

    images = torch.stack(images,0)
    leng = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions),max(leng)).long()

    for i,cap in enumerate(captions):
        end=leng[i]
        targets[i,:end]=cap[:end]

    return images,targets,leng



def get_loader(args):
    coco=Dataset(args)
    data_loader=torch.utils.data.DataLoader(dataset=coco,
                                            batch_size=args['batch_size'],
                                            shuffle=args['shuffle'],
                                            num_workers=args['num_workers'],
                                            drop_last=True,
                                            collate_fn=collate_fn)
    return data_loader

