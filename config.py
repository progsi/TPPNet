# coding:utf8
import warnings
import torch as t

class DefaultConfig(object):

    model = 'CQTTPPNet'  # 使用的模型，名字必须与models/__init__.py中的名字一致
    # model = 'CQT300Net'  # 使用的模型，名字必须与models/__init__.py中的名字一致
    feature = 'cqt'
    full_test = False # sh: to prevent use of all datasets
    load_model_path = None  # 加载预训练的模型的路径，为None代表不加载
    load_latest = False
    batch_size = 128  # batch size
    use_gpu = True # True  # user GPU or not
    num_workers = 4  # how many workers for loading data


    max_epoch = 100
    lr = 0.001  # initial learning rate
    lr_decay = 0.8  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-5  # 损失函数
    notes = None

    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        self.device =t.device('cuda') if self.use_gpu else t.device('cpu')

        # self.device =t.device('cuda') if t.cuda.is_available() else t.device('cpu')
        
        print('+------------------------------------------------------+')
        print('|','user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print('|',k, getattr(self, k))
        print('+------------------------------------------------------+')
opt = DefaultConfig()
