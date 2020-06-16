# Self-Supervised Human Depth Estimation from Monocular Videos

### Requirements
* Python 3.7
* [TensorFlow](https://www.tensorflow.org/) tested on version 1.14

#### Linux Setup with virtualenv
```
virtualenv self_human
source self_human/bin/activate
pip install -U pip
deactivate
source self_human/bin/activate
pip install -r requirements.txt
```
#### Install TensorFlow
With GPU:
```
pip install tensorflow-gpu==1.14
```
Without GPU:
```
pip install tensorflow==1.14
```

### Demo

1. Download the pre-trained models

2. predict base depth with finetuned hmr model
```
cd ./tracknet
python generate_tracknet_depth.py
```

3. predict detail depth
```
cd ./../reconnet/predict
python demo_tang_2019.py
```


### Citation
If you use this code for your research, please consider citing:
```
@inproceedings{tan2020self,
  title={Self-Supervised Human Depth Estimation from Monocular Videos},
  author={Tan, Feitong and Zhu, Hao and Cui, Zhaopeng and Zhu, Siyu and Pollefeys, Marc and Tan, Ping},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={650--659},
  year={2020}
}
```
