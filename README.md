# Image-Captioning
This project focuses on implementing a baseline model of image captioning, which is proposed in [show and tell:A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555). This baseline model is composed of two parts, one is CNN encoder part and the other part is RNN decoder part. During the implementation period, this project refers to another implementation in [pytorch tutorial](https://github.com/yunjey/pytorch-tutorial). However, in their implementation, they just gives main parts of this baseline without evaluation, beam search decoding and so on. Therefore, this project extends their work and gives a more comprehensive version of baseline model.

## Principle
The basic idea behind image captioning is that matching image representation and captioning representation in the same semantic space. Therefore, the first step is encoding image into a fixed length vector, and then this fixed length vector is used as input of RNN decoder. The process of encoding an image into a fixed length vector is completed by CNN which has broadly adopted to extract image features in many tasks. In original papaer, author adopted a CNN model pretrained on XXX dataset. However, in this project, a more advanced CNN model, `resnet152`, is used. Therefore, a improvement in performance is expected. 

## Getting started
### Prerequisizte
If you want to run this project without any problem, it is better recreate a new environment since there are some environment requirements for this experiment. This project requires `pytorch 0.4`, `tensorflow` and `cocoeval`. In order to make this process eaiser, a `requirement` is included in this repository. Run following command to install required environment.
```
conda-env create -n image_captioning -f requirement.yml
```
In addition to that, you also need to install [standard evaluation repository](https://github.com/tylin/coco-caption) in order to evaluate model's performance.

### Generate data
In original paper, standard dataset [Microsoft COCO](http://cocodataset.org/#home) is used, and eval dataset is divided to two parts, one is used for training, the other part is used for testing. Therefore, a script is required to rebuild training dataset and testing dataset. This can be done by running following command.
```
python split_dataset.py
```

## Train model
Default setting is given in `run.py` script. If you want to do any modification, you only need to modify corresponding parameters in this script. Once parameters are set, just run following command to start to train model.
```
python run.py
```

## Test model
### Evaluate performance with standard metrics
In original paper, three standard metrics are used to evaluate the performance of model. They are `BLEU-4`, `METEOR`, and `CIDEr` respectively. Note that there some tricks running evaluation scripts. Because evaluation repository is written with `python 2.7` and this project is written with `python 3.6`, you may need to generate captions and evaluate captions seperately. For example, once you have trained your model, you can run following command in `python 3.6` to generate captions from test dataset.
```
python test.py
```
and then switch to `python 2.7` to evalute generated captions with following command.
```
python evaluate.py
```

### Generate captions from specified images
If you want to test model on specific images, you need to put image in a specified folder and run following command.
```
python sample.py
```

## Author
* *ZVengin*

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

