
## Dependencies

Python 3.6.8

Pytorch 1.0.1


## Dataset

We conduct experiments on two benchmark multi-label classification datasets, i.e., VOC2007 and MS-COCO.

* Download [VOC2007] dataset.

* Download [MS-COCO] dataset.

Please go into folder ``` data/dataloader ``` and use our dataloader to create the dataset.


### Training

- Step1: train the classification model.

	- run ``` python main.py --mode 'train' --pretrain ```

- Step2: train the internal calssifier.

	- run ``` python main.py --mode 'train' --truncation 'layer0' ```

- If you want to train the internal calssifier on other internal layer, just replace the parameter like this:

	- run ``` python main.py --mode 'train' --truncation 'layer1' ```


### Test

- Step1: test the classification model.

	- run ``` python main.py --mode 'test' --checkpoint_name 'epoch_100_snapshot.pth' ```

- Step2: test the internal calssifier.

	- run ``` python main.py --mode 'test' --truncation 'layer0' --truncation_checkpoint_name 'epoch_100_snapshot.pth' ```