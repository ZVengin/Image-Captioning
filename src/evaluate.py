import os

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


def compute_score(annotation_file,fake_caption_file,score_file):
    coco=COCO(annotation_file)
    cocoRes=coco.loadRes(fake_caption_file)
    cocoEval=COCOEvalCap(coco,cocoRes)
    
    cocoEval.params['image_id']=cocoRes.getImgIds()

    cocoEval.evaluate()

    with open(score_file,'w') as f:
        for metric,score in cocoEval.eval.items():
            f.write('Metric: {}, Score: {}\n'.format(metric,str(score)))

if __name__=='__main__':
    exp_dir='../data_dir/exp_1'
    args = {
        'data_dir': '../data_dir',
        'exp_dir': exp_dir,
        'raw_data_dir': '../raw_data_dir/annotations',
        'caption_file': 'generated_captions.txt',
        'evaluation_file': 'evaluation_captions.json',
        'evaluation_result': 'evaluation_score.txt'}
    compute_score(os.path.join(args['raw_data_dir'], 'captions_val2014.json'),
                  os.path.join(args['exp_dir'], args['model_dir'], args['caption_file']),
                  os.path.join(args['exp_dir'], args['model_dir'], args['evaluation_result']))
