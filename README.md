# Collaborative Deep Reinforcement Learning

Collaborative Deep Reinforcement Learning (CDRL) is a collaborative framework to enable knowledge transfer among heterogeneous tasks.

### Dependencies

* Python 2.7
* [six](https://pypi.python.org/pypi/six) (for py2/3 compatibility)
* [TensorFlow](https://www.tensorflow.org/) 0.11
* [tmux](https://tmux.github.io/) (the start script opens up a tmux session with multiple windows)
* [htop](https://hisham.hm/htop/) (shown in one of the tmux windows)
* [gym](https://pypi.python.org/pypi/gym)
* gym[atari]
* [universe](https://pypi.python.org/pypi/universe)
* [opencv-python](https://pypi.python.org/pypi/opencv-python)
* [numpy](https://pypi.python.org/pypi/numpy)
* [scipy](https://pypi.python.org/pypi/scipy)

### Folder structure
|>**CDRL**</br>
>The code for experiments in paper section 5.4 Collaborative Deep Reinforcement Learning

|>**heterogeneousTransfer**</br>
>The code for experiments in paper section 5.3 Certificated Heterogeneous Transfer

|>**model** </br>
>The pre-trained model used for section heterogeneous transfer

### How to run:
|>**heterogeneousTransfer**</br>
>```python train.py --num-MTL-workers 8 0 --env-id PongDeterministic-v3_Bowling-v0```

|>**CDRL**</br>
>```python train.py --num-MTL-workers 8 8 --env-id PongDeterministic-v3_Bowling-v0```


### Paper: 
[Collaborative Deep Reinforcement Learning](https://128.84.21.199/abs/1702.05796) </br>
 Kaixiang Lin, Shu Wang and Jiayu Zhou</br> 

###Acknowledgement </br>
This code refer to [OpenAI universe starter agent](https://github.com/openai/universe-starter-agent). 

