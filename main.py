import os
import torch
from cqt_loader import *
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import models
from config import opt
from torchnet import meter
from tqdm import tqdm
import numpy as np
import time
import torch.nn.functional as F
import torch
import torch.nn as nn
from utility import *

def train(**kwargs):
    parallel = True
    opt.model, opt.notes = 'CQTTPPNet','single_train'
    opt.load_latest=False
    data_length = 400
    opt._parse(kwargs)
    # step1: configure model
    model = getattr(models, opt.model)()
    if parallel is True: 
        model = torch.nn.DataParallel(model)
    if parallel is True:
        if opt.load_latest is True:
            model.module.load_latest(opt.notes)
        elif opt.load_model_path:
            model.module.load(opt.load_model_path)
    else:
        if opt.load_latest is True:
            model.load_latest(opt.notes)
        elif opt.load_model_path:
            model.load(opt.load_model_path)
    model.to(opt.device)

    # step2: data

    train_data0 = CQT('train', out_length=data_length)
    val_data350 = CQT('songs350', out_length=None)
    val_data80 = CQT('songs80', out_length=None)
    val_data = CQT('val', out_length=None)
    test_data = CQT('test', out_length=None)
    train_dataloader0 = DataLoader(train_data0, opt.batch_size, shuffle=True,num_workers=opt.num_workers)
    val_dataloader350 = DataLoader(val_data350, 1, shuffle=False,num_workers=1)
    val_dataloader = DataLoader(val_data, 1, shuffle=False,num_workers=1)
    val_dataloader80 = DataLoader(val_data80, 1, shuffle=False, num_workers=1)
    test_dataloader = DataLoader(test_data, 1, shuffle=False,num_workers=1)

    #step3: criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    lr = opt.lr
    if parallel is True:
        optimizer = torch.optim.Adam(model.module.parameters(), lr=lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,mode='min',factor=opt.lr_decay,patience=1, verbose=True,min_lr=5e-6)
    #train
    best_MAP=0
    for epoch in range(opt.max_epoch):
        running_loss = 0
        num = 0
        for (data0, label0) in tqdm(train_dataloader0):
            data=data0
            label=label0
            # train model
            input = data.requires_grad_()
            input = input.to(opt.device)
            target = label.to(opt.device)
            optimizer.zero_grad()
            score, _ = model(input)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num += target.shape[0]
        running_loss /= num 
        print(running_loss)
        if parallel is True:
            model.module.save(opt.notes)
        else:
            model.save(opt.notes)
        # update learning rate
        scheduler.step(running_loss) 
        # validate
        MAP=0
        MAP += val_slow(model, val_dataloader350, epoch)
        MAP += val_slow(model, val_dataloader80, epoch)
        val_quick(model,val_dataloader)
        val_quick(model,test_dataloader)
        if MAP>best_MAP:
            best_MAP=MAP
            print('*****************BEST*****************')
        print('')
        model.train()


# multi_size train
def multi_train(**kwargs):
    parallel = True 
    opt.model = 'CQTTPPNet'
    opt.notes='Train_Val'
    opt.batch_size=32
    #opt.load_latest=True
    #opt.load_model_path = ''
    opt._parse(kwargs)
    # step1: configure model
    
    model = getattr(models, opt.model)() 
    if parallel is True: 
        model = torch.nn.DataParallel(model)
    if parallel is True:
        if opt.load_latest is True:
            model.module.load_latest(opt.notes)
        elif opt.load_model_path:
            model.module.load(opt.load_model_path)
    else:
        if opt.load_latest is True:
            model.load_latest(opt.notes)
        elif opt.load_model_path:
            model.load(opt.load_model_path)
    model.to(opt.device)
    print(model)
    # step2: data
   
    train_data0 = CQT('train', out_length=200)
    train_data1 = CQT('train', out_length=300)
    train_data2 = CQT('train', out_length=400)
    val_data350 = CQT('songs350', out_length=None)
    val_data80 = CQT('songs80', out_length=None)
    val_data = CQT('val', out_length=None)
    test_data = CQT('test', out_length=None)
    val_data2000 = CQT('songs2000', out_length=None)
    val_datatMazurkas = CQT('Mazurkas', out_length=None)
    train_dataloader0 = DataLoader(train_data0, opt.batch_size, shuffle=True,num_workers=opt.num_workers)
    train_dataloader1 = DataLoader(train_data1, opt.batch_size, shuffle=True,num_workers=opt.num_workers)
    train_dataloader2 = DataLoader(train_data2, opt.batch_size, shuffle=True,num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, 1, shuffle=False,num_workers=1)
    test_dataloader = DataLoader(test_data, 1, shuffle=False,num_workers=1)
    val_dataloader80 = DataLoader(val_data80, 1, shuffle=False, num_workers=1)
    val_dataloader350 = DataLoader(val_data350, 1, shuffle=False, num_workers=1)
    val_dataloader2000 = DataLoader(val_data2000, 1, shuffle=False, num_workers=1)
    val_dataloaderMazurkas = DataLoader(val_datatMazurkas,1, shuffle=False,num_workers=1)
    #step3: criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    lr = opt.lr
    if parallel is True:
        optimizer = torch.optim.Adam(model.module.parameters(), lr=lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,mode='min',factor=opt.lr_decay,patience=2, verbose=True,min_lr=5e-6)
    #train
    best_MAP=0
    val_slow(model, val_dataloader350, -1)
    #val_quick(model,val_dataloader)
    for epoch in range(opt.max_epoch):
        running_loss = 0
        num = 0
        for (data0, label0),(data1, label1),(data2, label2) in tqdm(zip(train_dataloader0, train_dataloader1, train_dataloader2)):
            for flag in range(3):
                if flag==0:
                    data=data0
                    label=label0
                elif flag==1:
                    data=data1
                    label=label1
                else:
                    data=data2
                    label=label2
                # train model
                input = data.requires_grad_()
                input = input.to(opt.device)
                target = label.to(opt.device)

                optimizer.zero_grad()
                score, _ = model(input)
                loss = criterion(score, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                num += target.shape[0]
        running_loss /= num 
        print(running_loss)
        if parallel is True:
            model.module.save(opt.notes)
        else:
            model.save(opt.notes)
        # update learning rate
        scheduler.step(running_loss) 
        # validate
        MAP=0
        MAP += val_slow(model, val_dataloader350, epoch)
        MAP += val_slow(model, val_dataloader80, epoch)
        val_slow(model, val_dataloader2000, epoch)
        val_slow(model, val_dataloaderMazurkas, epoch)
        #val_quick(model,val_dataloader)
        val_quick(model,test_dataloader)
        if MAP>best_MAP:
            best_MAP=MAP
            print('*****************BEST*****************')
        print('')
        model.train()

   
@torch.no_grad()
def multi_val_slow(model, dataloader1,dataloader2, epoch):
    model.eval()
    labels, features,features2 = None, None, None
    for ii, (data, label) in enumerate(dataloader1):
        input = data.to(opt.device)
        # print(input.shape)
        score, feature = model(input)
        feature = feature.data.cpu().numpy()
        label = label.data.cpu().numpy()
        if features is not None:
            features = np.concatenate((features, feature), axis=0)
            labels = np.concatenate((labels,label))
        else:
            features = feature
            labels = label
    for ii, (data, label) in enumerate(dataloader2):
        input = data.to(opt.device)
        # print(input.shape)
        score, feature = model(input)
        feature = feature.data.cpu().numpy()
        if features2 is not None:
            features2 = np.concatenate((features2, feature), axis=0)
        else:
            features2 = feature
            
    features = norm(features+features2)
    dis2d = get_dis2d4(features)
    
    if len(labels) == 350:
        MAP, top10, rank1 = calc_MAP(dis2d, labels,[100, 350])
    else :
        MAP, top10, rank1 = calc_MAP(dis2d, labels)
        MAP2 = compute_map(dis2d, labels)

    print(epoch, MAP, top10, rank1 )
    print(MAP2)
    model.train()
    return MAP

    
@torch.no_grad()
def val_slow(model, dataloader, epoch, ext_mode=False):
    model.eval()
    total, correct = 0, 0
    labels, features = None, None

    for ii, (data, label) in enumerate(dataloader):
        input = data.to(opt.device)
        # print(input.shape)
        score, feature = model(input)
        feature = feature.data.cpu().numpy()
        label = label.data.cpu().numpy()
        if features is not None:
            features = np.concatenate((features, feature), axis=0)
            labels = np.concatenate((labels,label))
        else:
            features = feature
            labels = label
    features = norm(features)
    #dis2d = get_dis2d4(features)
    dis2d = -np.matmul(features, features.T) # [-1,1] Because normalized, so mutmul is equal to ED
    np.save('dis80.npy',dis2d)
    np.save('label80.npy',labels)
    if len(labels) == 350:
        MAP, top10, rank1 = calc_MAP(dis2d, labels,[100, 350])
    #elif len(labels) == 160:    MAP, top10, rank1 = calc_MAP(dis2d, labels,[80, 160])
    else :
        MAP, top10, rank1 = calc_MAP(dis2d, labels)
        # MAP2 = compute_map(dis2d, labels)
        if ext_mode:
            ext = '-ext'
        else:
            ext = ''
        np.save(f"hpcp/10/dis2d{ext}.npy", dis2d)
        np.save(f"hpcp/10/labels{ext}.npy", labels)

    print(epoch, MAP, top10, rank1 )
    # print(MAP2)
    model.train()
    return MAP

    
@torch.no_grad()
def val_quick(model, dataloader,note=None):
    print('-----------------------------')
    model.eval()
    total, correct = 0, 0
    features, labels  = np.zeros([len(dataloader),300]), np.zeros(len(dataloader))
    #features, labels = None,None
    time1 = time.time()
    for ii, (data, label) in tqdm(enumerate(dataloader)):
        input = data.to(opt.device)

        score, feature = model(input)
        feature = feature.data.cpu().numpy()
        label = label.data.cpu().numpy()
        features[ii], labels[ii] = feature, label # only used when batch_size=1, otherwise use the code below

    features = norm(features)

    dis = np.matmul(features, features.T)
    path_dir = 'hpcp/10'
    if not os.path.exists(path_dir):
        os.mkdir(path_dir)
    features = features.astype(np.float32)
    features.tofile(path_dir+'/tmp_data.bin')
    np.savetxt(path_dir+'/tmp_ver.txt', labels, fmt='%d')
    thread_num=30
    if len(labels) == 350:
        os.system('multi_map/main_test %s 100 350 300 %d'% (path_dir,thread_num) )
    else:
        os.system('multi_map/main_test %s %d %d 300 %d' % (path_dir,len(labels), len(labels),thread_num))
    model.train()
    
def test(**kwargs):
    opt.batch_size=1
    opt.num_workers=1
    opt.model = 'CQTTPPNet'
    opt.load_latest = False
    opt.load_model_path = 'check_points/best.pth'
    opt._parse(kwargs)
    
    model = getattr(models, opt.model)() 
    # print(model)
    
    if opt.load_latest is True:
        model.load_latest(opt.notes)
    elif opt.load_model_path:
        model.load(opt.load_model_path, opt.device)
    
    model.to(opt.device)

    if not opt.full_test == True:
        test_data = CQT('shs-yt-1300', out_length=None)
        test_dataloader = DataLoader(test_data, 1, shuffle=False,num_workers=1)
        test_data_ext = CQT('shs-yt-1300-ext', out_length=None)
        test_dataloader_ext = DataLoader(test_data_ext, 1, shuffle=False,num_workers=1)
        val_slow(model, test_dataloader, 0)
        val_slow(model, test_dataloader_ext, 0, True)
    else:
        # val_data350 = CQT('songs350', out_length=None)
        val_data80 = CQT('songs80', out_length=None)
        val_data = CQT('val', out_length=None)
        test_data = CQT('test', out_length=None)
        val_data2000 = CQT('songs2000', out_length=None)
        val_datatMazurkas = CQT('Mazurkas', out_length=None)
        val_dataloader = DataLoader(val_data, 1, shuffle=False,num_workers=1)
        test_dataloader = DataLoader(test_data, 1, shuffle=False,num_workers=1)
        val_dataloader80 = DataLoader(val_data80, 1, shuffle=False, num_workers=1)
        # val_dataloader350 = DataLoader(val_data350, 1, shuffle=False, num_workers=1)
        val_dataloader2000 = DataLoader(val_data2000, 1, shuffle=False, num_workers=1)
        val_dataloaderMazurkas = DataLoader(val_datatMazurkas,1, shuffle=False,num_workers=1)
    
        # val_slow(model, val_dataloader350, 0)
        val_slow(model, val_dataloader80, 0)
        val_slow(model, val_dataloader2000,0)
        val_slow(model, val_dataloaderMazurkas, 0)
        val_quick(model, val_dataloader)
        val_quick(model, test_dataloader)
    
    
def multi_test(**kwargs):
    opt.batch_size=1
    opt.num_workers=1
    opt.model = 'CQTSPPNet10'
    opt.load_latest = False
    opt.load_model_path = None
    opt._parse(kwargs)
    model = getattr(models, opt.model)() 
    if opt.load_latest is True:
        model.load_latest(opt.notes)
    elif opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device)

    val_data = CQT('songs350', out_length=None)
    val_data80 = CQT('songs80', out_length=None)
    #val_datatest = CQT('test', out_length=None)
    val_data2000 = CQT('songs2000', out_length=None)
    val_datanew80 = CQT('new80', out_length=None)
    transf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(([84,300])),
            transforms.ToTensor(),
        ])
    _val_data = CQT('songs350', out_length=None,transform=transf)
    _val_data80 = CQT('songs80', out_length=None,transform=transf)
    val_dataloader = DataLoader(val_data, 1, shuffle=False,num_workers=opt.num_workers)
    val_dataloader80 = DataLoader(val_data80,1, shuffle=False,num_workers=opt.num_workers)
    _val_dataloader = DataLoader(_val_data, 1, shuffle=False,num_workers=opt.num_workers)
    _val_dataloader80 = DataLoader(_val_data80,1, shuffle=False,num_workers=opt.num_workers)
    
    multi_val_slow(model, val_dataloader,_val_dataloader, 0)
    multi_val_slow(model, val_dataloader80,_val_dataloader80, 0)

   


if __name__=='__main__':
    import fire
    fire.Fire()
