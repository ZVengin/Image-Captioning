import os
import json
import random

from pycocotools.coco import COCO

def split_dataset(raw_data_dir,dst_data_dir,size_list):
    valid_size,test_size=size_list[0],size_list[1]
    valid_idx_list,test_idx_ist=list(),list()
    valid_ann_idx,test_img_idx=list(),list()

    coco=COCO(os.path.join(raw_data_dir,'annotations','captions_val2014.json'))
    image_idx_list=list(coco.imgs.keys())
    valid_size=len(image_idx_list)-test_size
    print('valid_size: {}\n test_size: {}'.format(valid_size,test_size))

    while len(valid_idx_list)<valid_size:
        idx=random.randint(0,len(image_idx_list)-1)
        if idx not in valid_idx_list:
            valid_idx_list.append(idx)
            valid_ann_idx+=[ann['id'] for ann in coco.imgToAnns[image_idx_list[idx]]]


    while len(test_idx_ist)<test_size:
        idx=random.randint(0,len(image_idx_list)-1)
        if idx not in valid_idx_list and idx not in test_idx_ist:
            test_idx_ist.append(idx)
            test_img_idx+=[image_idx_list[idx]]

#    valid_ann_path=os.path.join(dst_data_dir,'validate_annotations.json')
    test_img_path=os.path.join(dst_data_dir,'test_images.json')
    train_ann_path=os.path.join(dst_data_dir,'train_annotations.json')

#    with open(valid_ann_path,'w')as f:
#        json.dump(valid_ann_idx,f)

    with open(test_img_path,'w') as f:
        json.dump(test_img_idx,f)

    train_coco=COCO(os.path.join(raw_data_dir,'annotations','captions_train2014.json'))
    train_ann_idx=list(train_coco.anns.keys())+valid_ann_idx
    print('train_annotation_length:{}'.format(len(train_ann_idx)))
    with open(train_ann_path,'w') as f:
        json.dump(train_ann_idx,f)

if __name__=='__main__':
    split_dataset('../raw_data_dir','../data_dir/exp_6',[0,4000])
