import numpy as np
import os
import h5py
import json
import matplotlib.pyplot as plt

##############################################################################

from builtins import range
import urllib.request, urllib.error, urllib.parse, os, tempfile

import numpy as np
from scipy.misc import imread, imresize




SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SQUEEZENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(img):
    """Preprocess an image for squeezenet.

    Subtracts the pixel mean and divides by the standard deviation.
    """
    return (img.astype(np.float32) / 255.0 - SQUEEZENET_MEAN) / SQUEEZENET_STD


def deprocess_image(img, rescale=False):
    """Undo preprocessing on an image and convert back to uint8."""
    img = (img * SQUEEZENET_STD + SQUEEZENET_MEAN)
    if rescale:
        vmin, vmax = img.min(), img.max()
        img = (img - vmin) / (vmax - vmin)
    return np.clip(255 * img, 0.0, 255.0).astype(np.uint8)


def image_from_url(url):
    """
    Read an image from a URL. Returns a numpy array with the pixel data.
    We write the image to a temporary file then read it back. Kinda gross.
    """
    try:
        f = urllib.request.urlopen(url)
        _, fname = tempfile.mkstemp()
        with open(fname, 'wb') as ff:
            ff.write(f.read())
        img = imread(fname)
        # os.remove(fname)
        return img
    except urllib.error.URLError as e:
        print('URL Error: ', e.reason, url)
    except urllib.error.HTTPError as e:
        print('HTTP Error: ', e.code, url)


def load_image(filename, size=None):
    """Load and resize an image from disk.

    Inputs:
    - filename: path to file
    - size: size of shortest dimension after rescaling
    """
    img = imread(filename)
    if size is not None:
        orig_shape = np.array(img.shape[:2])
        min_idx = np.argmin(orig_shape)
        scale_factor = float(size) / orig_shape[min_idx]
        new_shape = (orig_shape * scale_factor).astype(int)
        img = imresize(img, scale_factor)
    return img

##############################################################################

BASE_DIR = './dataset/coco_captioning'

def load_coco_data(base_dir=BASE_DIR,
                   max_train=None,
                   pca_features=True):
    data = {}
    caption_file = os.path.join(base_dir, 'coco2014_captions.h5')
    with h5py.File(caption_file, 'r') as f:
        for k, v in f.items():
            data[k] = np.asarray(v)

    if pca_features:
        train_feat_file = os.path.join(base_dir, 'train2014_vgg16_fc7_pca.h5')
    else:
        train_feat_file = os.path.join(base_dir, 'train2014_vgg16_fc7.h5')
    with h5py.File(train_feat_file, 'r') as f:
        data['train_features'] = np.asarray(f['features'])

    if pca_features:
        val_feat_file = os.path.join(base_dir, 'val2014_vgg16_fc7_pca.h5')
    else:
        val_feat_file = os.path.join(base_dir, 'val2014_vgg16_fc7.h5')
    with h5py.File(val_feat_file, 'r') as f:
        data['val_features'] = np.asarray(f['features'])

    dict_file = os.path.join(base_dir, 'coco2014_vocab.json')
    with open(dict_file, 'r') as f:
        dict_data = json.load(f)
        for k, v in dict_data.items():
            data[k] = v

    train_url_file = os.path.join(base_dir, 'train2014_urls.txt')
    with open(train_url_file, 'r') as f:
        train_urls = np.asarray([line.strip() for line in f])
    data['train_urls'] = train_urls

    val_url_file = os.path.join(base_dir, 'val2014_urls.txt')
    with open(val_url_file, 'r') as f:
        val_urls = np.asarray([line.strip() for line in f])
    data['val_urls'] = val_urls

    # Maybe subsample the training data
    if max_train is not None:
        num_train = data['train_captions'].shape[0]
        mask = np.random.randint(num_train, size=max_train)
        data['train_captions'] = data['train_captions'][mask]
        data['train_image_idxs'] = data['train_image_idxs'][mask]

    return data


def decode_captions(captions, idx_to_word):
    singleton = False
    if captions.ndim == 1:
        singleton = True
        captions = captions[None]
    decoded = []
    N, T = captions.shape
    for i in range(N):
        words = []
        for t in range(T):
            word = idx_to_word[captions[i, t]]
            if word != '<NULL>':
                words.append(word)
            if word == '<END>':
                break
        decoded.append(' '.join(words))
    if singleton:
        decoded = decoded[0]
    return decoded


def sample_coco_minibatch(data, batch_size=100, split='train'):
    split_size = data['%s_captions' % split].shape[0]
    mask = np.random.choice(split_size, batch_size)
    captions = data['%s_captions' % split][mask]
    image_idxs = data['%s_image_idxs' % split][mask]
    image_features = data['%s_features' % split][image_idxs]
    urls = data['%s_urls' % split][image_idxs]
    return captions, image_features, urls

##############################################################################
def affine_forward(x,w,b):
    '''
    计算仿射(完全连接)图层的前向传播。

    :param x: 输入特征 (N,d_1,...,d_k)
    :param w: 权重 (D,M)
    :param b: 偏置 (M,)
    :return:
    :param out: 输出, of shape (N, M)
    :param cache: (x, w, b)
    '''
    out=x.reshape(x.shape[0],-1).dot(w)+b
    cache=(x,w,b)
    return out,cache

def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx = dout.dot(w.T).reshape(x.shape)
    dw = x.reshape(x.shape[0], -1).T.dot(dout)
    db = np.sum(dout, axis=0)
    return dx, dw, db

def word_embedding_forward(x, W):
    '''
    词嵌入的前向传播。
    :param x:
    :param W:
    :return:
    '''
    out,cache=None,None
    out=W[x,:]
    cache=(x,W.shape)
    return out,cache

def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    x, w_shape = cache
    dW = np.zeros(w_shape)  # (V, D)

    np.add.at(dW, x, dout)

    return dW

def rnn_step_forward(x,prev_h,Wx,Wh,b):
    '''
    运行单个时间步长的RNN前向传播，使用tanh作为激活函数

    :param x:当前时间的输入数据（N，D）
    :param prev_h:上一时刻的影藏状态，（N，H）
    :param Wx:输入到隐藏层的权重，（D，H）
    :param Wh:隐藏层到隐藏层的权重，（H，H）
    :param b:偏置，（H,）
    :return:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    '''
    next_h,cache=None,None

    affine_h=np.dot(prev_h,Wh) #(N,H)
    affine_x=np.dot(x,Wx)+b #(N,H)
    new_h=affine_x+affine_h #(N,H)
    next_h=np.tanh(new_h)
    cache=(x,prev_h,Wx,Wh,next_h)

    return next_h,cache

def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state, of shape (N, H)
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None

    x, prev_h, Wx, Wh, next_h = cache
    d_new_h = dnext_h * (1 - next_h ** 2)
    dWx = np.dot(x.T, d_new_h)
    db = np.sum(d_new_h, axis=0)
    dx = np.dot(d_new_h, Wx.T)
    dWh = np.dot(prev_h.T, d_new_h)
    dprev_h = np.dot(d_new_h, Wh.T)

    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x,h0,Wx,Wh,b):
    '''
    在整个数据序列上运行RNN前向传播。

    :param x:输入整个时间序列的数据 （N,T,D）
    :param h0:初始隐藏状态 (N,H)
    :param Wx:输入到隐藏层的权重，（D，H）
    :param Wh:隐藏层到隐藏层的权重，（H，H）
    :param b:偏置，（H,）
    :return:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    '''
    h,cache=None,None

    N,T,D=x.shape
    hiddens=[]
    hidden=h0

    for i in range(T):
        xt=x[:,i,:]
        hidden,_=rnn_step_forward(xt,hidden,Wx,Wh,b)
        hiddens.append(hidden)
    h=np.stack(hiddens,axis=1)
    cache=(x,h0,Wh,Wx,b,h)

    return h,cache

def rnn_backward(dh,cache):
    '''
    计算RNN网络的反向传播

    :param dh: 上一层的梯度，（N,T,H）
    :param cache:
    :return:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    '''
    dx, dh0, dWx, dWh, db = None, None, None, None, None

    x, h0, Wh, Wx, b, h = cache
    _,T,_=dh.shape
    dx=np.zeros_like(x)
    dWx=np.zeros_like(Wx)
    dWh=np.zeros_like(Wh)
    db = np.zeros_like(b)
    dprev_h = 0

    for i in range(T):
        t=T-1-i
        xt=x[:,t,:]
        dht=dh[:,t,:]
        if t>0:
            prev_h=h[:,t-1,:]
        else:
            prev_h=h0
        next_h=h[:,t,:]

        dx[:,t,:],dprev_h,dwx, dwh, db_=rnn_step_backward(dht + dprev_h, (xt, prev_h, Wx, Wh, next_h))
        dWx += dwx
        dWh += dwh
        db += db_
        dh0 = dprev_h

    return dx, dh0, dWx, dWh, db

def temporal_affine_forward(x, w, b):
    '''
    前向传播将每一个时间的输出转化为最终预测的结果

    :param x:输入数据，（N，T,D）
    :param w:（D，M）,M为词汇表中的单词个数
    :param b:(M,)
    :return:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    '''
    N,T,D=x.shape
    M=b.shape[0]
    out=x.reshape(N*T,D).dot(w).reshape(N,T,M)+b
    cache=x,w,b,out
    return out,cache

def temporal_affine_backward(dout,cache):
    '''
    时间仿射层的反向传播。

    :param dout:输出梯度，（N，T，M）
    :param cache:前向传播数据
    :return:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    '''
    x,w,b,out=cache
    N,T,D=x.shape
    M=b.shape[0]

    dx=dout.reshape(N*T,M).dot(w.T).reshape(N,T,D)
    dw=dout.reshape(N*T,M).T.dot(x.reshape(N*T,D)).T
    db=dout.sum(axis=(0,1))

    return dx,dw,db


def temporal_softmax_loss(x, y, mask, verbose=False):
    '''
    在RNN中使用softmax_loss作为损失函数。
    我们假设我们在一个大小为V的词汇表上，预测得到一个长度为T时间序列的输出。输入x是词汇表
    在所有时间步长上的预测分数。y为每个时间步长上的真实标签，对于每一个时间预测采用交叉熵
    作为每一步的损失，然后对这些损失求和取平均得到最终的损失。

    :param x:输出分数，（N,T,V）
    :param y:标签下标，（N，T）
    :param mask:
    :param verbose:
    :return:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    '''
    N,T,V=x.shape

    x_flat=x.reshape(N*T,V)
    y_flat=y.reshape(N*T)
    mask_flat=mask.reshape(N*T)

    probs=np.exp(x_flat-np.max(x_flat,axis=1,keepdims=True))
    probs/=np.sum(probs,axis=1,keepdims=True)
    loss=-np.sum(mask_flat*np.log(probs[np.arange(N*T),y_flat]))/N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx

##############################################################################

class RNN():
    def __init__(self,word_to_idx,input_dim=512,wordvec_dim=128,
                 hidden_dim=128,cell_type='rnn',dtype=np.float32):
        '''

        :param world_to_idx:字典，存储了单词到整数的映射关系.
        :param input_dim:单个输入的维度
        :param wordvec_dim:单词向量的维度.
        :param hidden_dim:隐层的大小.
        :param cell_type:'rnn' or 'lstm'.
        :param dtype:
        '''
        if cell_type not in {'rnn', 'lstm'}:
            raise ValueError('Invalid cell type "%s"' % cell_type)

        self.cell_type=cell_type
        self.dtype = dtype
        self.word_to_idx = word_to_idx
        self.idx_to_word={i:w for w,i in self.word_to_idx.items()}
        self.params = {}

        vocab_size=len(word_to_idx)

        self._null=word_to_idx['<NULL>']
        self._start=word_to_idx.get('<START>',None)
        self._end=word_to_idx.get('<END>',None)

        # 初始化代表每个单词的向量
        self.params['W_embed']=np.random.randn(vocab_size,wordvec_dim)
        self.params['W_embed']/=100

        self.params['W_proj'] = np.random.randn(input_dim, hidden_dim)
        self.params['W_proj'] /= np.sqrt(input_dim)
        self.params['b_proj'] = np.zeros(hidden_dim)

        # 初始化RNN的参数
        dim_mul = {'lstm': 4, 'rnn': 1}[cell_type]
        self.params['Wx'] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)
        self.params['Wx'] /= np.sqrt(wordvec_dim)
        self.params['Wh'] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
        self.params['Wh'] /= np.sqrt(hidden_dim)
        self.params['b'] = np.zeros(dim_mul * hidden_dim)

        # 得到每个时间点的输出
        self.params['W_vocab'] = np.random.randn(hidden_dim, vocab_size)
        self.params['W_vocab'] /= np.sqrt(hidden_dim)
        self.params['b_vocab'] = np.zeros(vocab_size)

        # 强制转换，保证每个权重矩阵都是我们设置的数据类型
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)

    def loss(self,feature,captions):
        '''
        返回loss和梯度（如果是训练过程）
        :param feature:输入的特征，shape (N, D)
        :param captions:Ground-truth; 数组，shape (N, T)，N张图片都用T个单词描述
        :return:
        '''
        # 输入的描述和输出的描述
        # 具体在于输入去掉最后一个单词'<end>' 输出去掉第一个单词'<start>'
        # 因为输入第一个单词的时候就已经需要预测下一个单词了 所以有一个单词的错位
        captions_in=captions[:,:-1]
        captions_out=captions[:,1:]

        # 每个描述长度有所不同，短的用NULL补齐到T长度，所以NULL不计入loss
        mask = (captions_out != self._null)

        # 将输入特征转化为RNN输入的参数
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']

        # embedding矩阵
        W_embed = self.params['W_embed']

        # 输入到隐藏层矩阵, 隐藏到隐藏层矩阵, 偏置矩阵
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']

        # 将输出转化为词向量的矩阵.
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        # 前向计算过程：
        # （1）使用仿射函数将从CNN提取的特征转换到第一个RNN的隐层状态(N,H)
        # （2）将captions_in从单词转换成向量（N，T，W）
        # （3）在RNN各个隐层中进行运算，（N，T，H）
        # （4）在每个时间点计算各个单词的得分（N，T，V）
        # （5）用softmax计算各时间点的loss
        # W表示单词向量的维度 V表示单词词库的个数
        loss, grads = 0.0, {}

        h0,affine_cache=affine_forward(feature,W_proj,b_proj) #[N,H]
        x,embed_cache=word_embedding_forward(captions_in,W_embed)#标签的词嵌入

        if self.cell_type=='rnn':
            h,rnn_cache=rnn_forward(x,h0,Wx,Wh,b)

        out,temp_affine_cache=temporal_affine_forward(h,W_vocab,b_vocab)
        loss,dout=temporal_softmax_loss(out,captions_out,mask)

        #反向传播，# 计算梯度
        dout,dw,db=temporal_affine_backward(dout,temp_affine_cache)
        grads['W_vocab']=dw
        grads['b_vocab'] = db

        if self.cell_type=="rnn":
            dout, dh0, dWx, dWh, db=rnn_backward(dout,rnn_cache)
        grads['Wx'] = dWx
        grads['Wh'] = dWh
        grads['b'] = db

        dw_embed = word_embedding_backward(dout, embed_cache)
        grads['W_embed'] = dw_embed

        _, dw_proj, db_proj = affine_backward(dh0, affine_cache)
        grads['W_proj'] = dw_proj
        grads['b_proj'] = db_proj

        return loss,grads

    def sample(self,features,max_length=30):
        '''
        执行网络的前向传播，通过输入特征获取输出字符

        <start>当作第一个时刻的输入，可以得到第一个时刻的输出，
        把第一个时刻的输出当作第二个时刻的输入,依次类推.
        :param feature: 输入数据，（N，D）
        :param max_length: 生成结果的最大长度
        :return:
        '''
        N=features.shape[0]
        captions = self._null * np.ones((N, max_length), dtype=np.int32)

        # Unpack parameters
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
        W_embed = self.params['W_embed']
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        h, _ = affine_forward(features, W_proj, b_proj)
        inputs, _ = word_embedding_forward(self._start, W_embed)
        # inputs = np.tile(inputs, (2, 1))
        c = np.zeros_like(h)
        for i in range(max_length):
            if self.cell_type == 'rnn':
                next_h, _ = rnn_step_forward(inputs, h, Wx, Wh, b)

            output, _ = temporal_affine_forward(next_h[:, np.newaxis, :], W_vocab, b_vocab)
            output_idx = np.argmax(output, axis=2)
            for j in range(output_idx.shape[0]):
                captions[j][i] = output_idx[j]
            inputs, _ = word_embedding_forward(output_idx[:, 0], W_embed)
            h = next_h

        return captions

##############################################################################

def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.

    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config


def adam(x, dx, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(x))
    config.setdefault('v', np.zeros_like(x))
    config.setdefault('t', 0)

    next_x = None
    beta1, beta2, eps = config['beta1'], config['beta2'], config['epsilon']
    t, m, v = config['t'], config['m'], config['v']
    m = beta1 * m + (1 - beta1) * dx
    v = beta2 * v + (1 - beta2) * (dx * dx)
    t += 1
    alpha = config['learning_rate'] * np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
    x -= alpha * (m / (np.sqrt(v) + eps))
    config['t'] = t
    config['m'] = m
    config['v'] = v
    next_x = x

    return next_x, config

##############################################################################
class CaptioningSolver(object):
    """基于RNN结构的训练过程"""
    def __init__(self,model, data, **kwargs):
        '''
        必选参数:
        - model: 符合一定要求的model
        - data: 特定格式的训练和验证数据

        可选参数:
        - update_rule: 可选梯度下降方法，默认sgd
        - optim_config: 梯度下降的参数
        - lr_decay: 学习率衰减参数，各个epoch更新lr_decay=lr*lr_decay.默认不衰减，即1.0
        - batch_size: 批大小
        - num_epochs: 训练的epoch数量
        - print_every: 几个epochs打印一次信息,默认一个.
        - verbose: 训练过程中是否打印信息
        '''
        self.model = model
        self.data = data

        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 128)
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.print_every = kwargs.pop('print_every', 1)
        self.verbose = kwargs.pop('verbose', True)

        # 有错误或者多余参数 报错
        if len(kwargs) > 0:
            extra = ','.join('"%s"' % k for k in kwargs.keys())
            raise ValueError('Unrecognized arguments %s' % extra)

        # if not hasattr(optim, self.update_rule):
        #     raise ValueError('Invalid update rule %s' % self.update_rule)
        #
        # self.update_rule = getattr(optim, self.update_rule)

        self.update_rule = adam

        self._reset()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # Make a deep copy of the optim_config for each parameter
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d

    def _step(self):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.
        """
        # Make a minibatch of training data
        minibatch = sample_coco_minibatch(self.data,
                                          batch_size=self.batch_size,
                                          split='train')
        captions, features, urls = minibatch

        # Compute loss and gradient
        loss, grads = self.model.loss(features, captions)
        # self.acc=self.check_accuracy(features,captions,batch_size=self.batch_size)
        self.loss_history.append(loss)

        # Perform a parameter update
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config

    def check_accuracy(self, X, y, num_samples=None, batch_size=100):
        """
        Check accuracy of the model on the provided data.

        Inputs:
        - X: Array of data, of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,)
        - num_samples: If not None, subsample the data and only test the model
          on num_samples datapoints.
        - batch_size: Split X and y into batches of this size to avoid using too
          much memory.

        Returns:
        - acc: Scalar giving the fraction of instances that were correctly
          classified by the model.
        """
        # return 0.0

        # Maybe subsample the data
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        # Compute predictions in batches
        num_batches = N / batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)

        return acc

    def train(self):
        """
        Run optimization to train the model.
        """
        num_train = self.data['train_captions'].shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in range(num_iterations):
            self._step()

            # Maybe print training loss
            if self.verbose and t % self.print_every == 0:
                print('(Iteration %d / %d) loss: %f' % (
                    t + 1, num_iterations, self.loss_history[-1]))


            # At the end of every epoch, increment the epoch counter and decay the
            # learning rate.
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]['learning_rate'] *= self.lr_decay


if __name__=="__main__":
    np.random.seed(231)

    data = load_coco_data(pca_features=True)

    # # Print out all the keys and values from the data dictionary
    # for k, v in data.items():
    #     if type(v) == np.ndarray:
    #         print(k, type(v), v.shape, v.dtype)
    #     else:
    #         print(k, type(v), len(v))
    #
    small_data = load_coco_data(max_train=50)

    small_rnn_model = RNN(
        cell_type='rnn',
        word_to_idx=data['word_to_idx'],
        input_dim=data['train_features'].shape[1],
        hidden_dim=512,
        wordvec_dim=256,
    )

    small_rnn_solver = CaptioningSolver(small_rnn_model, small_data,
                                        update_rule='adam',
                                        num_epochs=100,
                                        batch_size=10,
                                        optim_config={
                                            'learning_rate': 5e-3,
                                        },
                                        lr_decay=0.95,
                                        verbose=True, print_every=10,
                                        )

    small_rnn_solver.train()

    # Plot the training losses
    plt.plot(small_rnn_solver.loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training loss history')
    plt.show()

    for split in ['train', 'val']:
        minibatch = sample_coco_minibatch(small_data, split=split, batch_size=2)
        gt_captions, features, urls = minibatch
        gt_captions = decode_captions(gt_captions, data['idx_to_word'])

        sample_captions = small_rnn_model.sample(features)
        sample_captions = decode_captions(sample_captions, data['idx_to_word'])

        for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
            plt.imshow(image_from_url(url))
            plt.title('%s\n%s\nGT:%s' % (split, sample_caption, gt_caption))
            plt.axis('off')
            plt.show()