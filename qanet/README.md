# QANet
A Tensorflow implementation of Google's [QANet](https://openreview.net/pdf?id=B14TlG-RW) (previously Fast Reading Comprehension (FRC)) from [ICLR2018](https://openreview.net/forum?id=B14TlG-RW). (Note: This is not an official implementation from the authors of the paper)

I wrote a blog post about implementing QANet. Check out [here](https://medium.com/@minsangkim/implementing-question-answering-networks-with-cnns-5ae5f08e312b) for more information!

Training and preprocessing pipeline have been adopted from [R-Net by HKUST-KnowComp](https://github.com/HKUST-KnowComp/R-Net). Demo mode is working. After training, just use `python config.py --mode demo` to run an interactive demo server.

Due to a memory issue, a single head dot-product attention is used as opposed to a 8 heads multi-head attention like in the original paper. The hidden size is also reduced to 96 from 128 due to usage of a GTX1080 compared to a P100 used in the paper. (8GB of GPU memory is insufficient. If you have a 12GB memory GPU please share your training results with us.)

Currently, the best model reaches EM/F1 = 70.8/80.1 in 60k steps (6~8 hours). Detailed results are listed below.

![Alt text](/../master/screenshots/figure.png?raw=true "Network Outline")

## Dataset
The dataset used for this task is [Stanford Question Answering Dataset](https://rajpurkar.github.io/SQuAD-explorer/).
Pretrained [GloVe embeddings](https://nlp.stanford.edu/projects/glove/) obtained from common crawl with 840B tokens used for words.

## Requirements
  * Python>=2.7
  * NumPy
  * tqdm
  * TensorFlow>=1.5
  * spacy==2.0.9
  * bottle (only for demo)

## Usage
To download and preprocess the data, run

```bash
# download SQuAD and Glove
sh download.sh
# preprocess the data
python config.py --mode prepro
```

Just like [R-Net by HKUST-KnowComp](https://github.com/HKUST-KnowComp/R-Net), hyper parameters are stored in config.py. To debug/train/test/demo, run

```bash
python config.py --mode debug/train/test/demo
```

To evaluate the model with the official code, run
```bash
python evaluate-v1.1.py ~/data/squad/dev-v1.1.json train/{model_name}/answer/answer.json
```

The default directory for the tensorboard log file is `train/{model_name}/event`

### Run in Docker container (optional)
To build the Docker image (requires nvidia-docker), run

```
nvidia-docker build -t tensorflow/qanet .
```

Set volume mount paths and port mappings (for demo mode)

```
export QANETPATH={/path/to/cloned/QANet}
export CONTAINERWORKDIR=/home/QANet
export HOSTPORT=8080
export CONTAINERPORT=8080
```

bash into the container
```
nvidia-docker run -v $QANETPATH:$CONTAINERWORKDIR -p $HOSTPORT:$CONTAINERPORT -it --rm tensorflow/qanet bash
```

Once inside the container, follow the commands provided above starting with downloading the SQuAD and Glove datasets.

### Pretrained Model
Pretrained model weights are temporarily not available.

## Detailed Implementaion

  * The model adopts character level convolution - max pooling - highway network for input representations similar to [this paper by Yoon Kim](https://arxiv.org/pdf/1508.06615.pdf).
  * The encoder consists of positional encoding - depthwise separable convolution - self attention - feed forward structure with layer norm in between.
  * Despite the original paper using 200, we observe that using a smaller character dimension leads to better generalization.
  * For regularization, a dropout of 0.1 is used every 2 sub-layers and 2 blocks.
  * Stochastic depth dropout is used to drop the residual connection with respect to increasing depth of the network as this model heavily relies on residual connections.
  * Query-to-Context attention is used along with Context-to-Query attention, which seems to improve the performance more than what the paper reported. This may be due to the lack of diversity in self attention due to 1 head (as opposed to 8 heads) which may have repetitive information that the query-to-context attention contains.
  * Learning rate increases from 0.0 to 0.001 in the first 1000 steps in inverse exponential scale and fixed to 0.001 from 1000 steps.
  * At inference, this model uses shadow variables maintained by the exponential moving average of all global variables.
  * This model uses a training / testing / preprocessing pipeline from [R-Net](https://github.com/HKUST-KnowComp/R-Net) for improved efficiency.

## Results
Here are the collected results from this repository and the original paper.

|      Model     | Training Steps | Size | Attention Heads | Data Size (aug) |  EM  |  F1  |
|:--------------:|:--------------:|:----:|:---------------:|:---------------:|:----:|:----:|
|       My model |     35,000     |  96  |        1        |   87k (no aug)  | 69.0 | 78.6 |
|       My model |     60,000     |  96  |        1        |   87k (no aug)  | 70.4 | 79.6 |
|       My model ( reported by [@jasonbw](https://github.com/jasonwbw))|     60,000     |  128  |        1        |   87k (no aug)  | 70.7 | 79.8 |
|       My model ( reported by [@chesterkuo](https://github.com/chesterkuo))|     60,000     |  128  |        8        |   87k (no aug)  | 70.8 | 80.1 |
| Original Paper |     35,000     |  128 |        8        |   87k (no aug)  |  NA  | 77.0 |
| Original Paper |     150,000    |  128 |        8        |   87k (no aug)  | 73.6 | 82.7 |
| Original Paper |     340,000    |  128 |        8        |    240k (aug)   | 75.1 | 83.8 |

## TODO's
- [x] Training and testing the model
- [x] Add trilinear function to Context-to-Query attention
- [x] Apply dropouts + stochastic depth dropout
- [x] Query-to-context attention
- [x] Realtime Demo
- [ ] Data augmentation by paraphrasing
- [ ] Train with full hyperparameters (Augmented data, 8 heads, hidden units = 128)

## Tensorboard
Run tensorboard for visualisation.
```shell
$ tensorboard --logdir=./
```
