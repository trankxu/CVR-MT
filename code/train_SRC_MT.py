import os
import sys
# from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from networks.models import DenseNet121,DenseNet161
from utils import losses, ramps
from utils.metrics import compute_AUCs
from utils.metric_logger import MetricLogger
from dataloaders import  dataset
from dataloaders.dataset import TwoStreamBatchSampler
from utils.util import get_timestamp
from validation import epochVal, epochVal_metrics, epochVal_metrics_test
from torchviz import make_dot

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/data1/bitgroup/trankxu/code/SRC-MT/data/skin/ISIC2018_Task3_Training_Input', help='dataset root dir')
parser.add_argument('--csv_file_train', type=str, default='/data1/bitgroup/trankxu/code/RAC-MT/data/skin/training.csv', help='training set csv file')
parser.add_argument('--csv_file_val', type=str, default='/data1/bitgroup/trankxu/code/RAC-MT/data/skin/validation.csv', help='validation set csv file')
parser.add_argument('--csv_file_test', type=str, default='/data1/bitgroup/trankxu/code/RAC-MT/data/skin/testing.csv', help='testing set csv file')
parser.add_argument('--exp', type=str,  default='xxxx', help='model_name')
parser.add_argument('--epochs', type=int,  default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default= 64, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default= 8, help='number of labeled data per batch')
parser.add_argument('--drop_rate', type=int, default=0.2, help='dropout rate')
parser.add_argument('--ema_consistency', type=int, default=1, help='whether train baseline model')
parser.add_argument('--labeled_num', type=int, default=1400, help='number of labeled')
parser.add_argument('--base_lr', type=float,  default=1e-4, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=2023, help='random seed')
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
### tune
parser.add_argument('--resume', type=str,  default=None, help='model to resume')
# parser.add_argument('--resume', type=str,  default=None, help='GPU to use')
parser.add_argument('--start_epoch', type=int,  default=0, help='start_epoch')
parser.add_argument('--global_step', type=int,  default=0, help='global_step')
### costs
parser.add_argument('--label_uncertainty', type=str,  default='U-Ones', help='label type')
parser.add_argument('--consistency_relation_weight', type=int,  default=1, help='consistency relation weight')
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="kl", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=30, help='consistency_rampup')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
base_lr = args.base_lr
labeled_bs = args.labeled_bs * len(args.gpu.split(','))

if args.deterministic:
    cudnn.benchmark = False#当这个参数设置为True时，启动算法的前期会比较慢，但算法跑起来以后会非常快。如果输入大小是固定的，采用cudnn.benchmark是可以加速推理和训练的。除了启动时会花额外的时间配置。但对于输入大小会变动的，每变一次就会配置一次，反而会影响速度

    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242

    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


if __name__ == "__main__":
    ## make logging file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
        os.makedirs(snapshot_path + './checkpoint')
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model(ema=False):
        # Network definition
        net = DenseNet121(out_size=dataset.N_CLASSES, mode=args.label_uncertainty, drop_rate=args.drop_rate)
        if len(args.gpu.split(',')) > 1:
            net = torch.nn.DataParallel(net)
        model = net.to("cuda:"+args.gpu)
        if ema:
            for param in model.parameters():# 返回一个生成器
                param.detach_()#不进行梯度更新
        return model

    model = create_model()
    ema_model = create_model(ema=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, 
                                 betas=(0.9, 0.999), weight_decay=5e-4)

    if args.resume:
        assert os.path.isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
        logging.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        args.global_step = checkpoint['global_step']
        # best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logging.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    # dataset
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    
    weak_trans = transforms.Compose([transforms.Resize((224, 224)),
                        transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                        transforms.RandomHorizontalFlip(),
                                                # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                                                # transforms.RandomRotation(10),
                                                # transforms.RandomResizedCrop(224),
                        transforms.ToTensor(),
                        normalize,
                        ])
                        
    strong_trans =  transforms.Compose([transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),     
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
            ])
            
    train_dataset = dataset.CheXpertDataset(root_dir=args.root_path,
                                            csv_file=args.csv_file_train,
                                            #######Chuankai Change#############
                                            transform=dataset.TransTwoStreams(dataset.TransformTwice(weak_trans),dataset.TransformTwice(weak_trans)))

    val_dataset = dataset.CheXpertDataset(root_dir=args.root_path,
                                          csv_file=args.csv_file_val,
                                          transform=transforms.Compose([
                                              transforms.Resize((224, 224)),
                                              transforms.ToTensor(),
                                              normalize,
                                          ]))
    test_dataset = dataset.CheXpertDataset(root_dir=args.root_path,
                                          csv_file=args.csv_file_test,
                                          transform=transforms.Compose([
                                              transforms.Resize((224, 224)),
                                              transforms.ToTensor(),
                                              normalize,
                                          ]))

    labeled_idxs = list(range(args.labeled_num))
    unlabeled_idxs = list(range(args.labeled_num, 7000))# 1:5
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)

    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    train_dataloader = DataLoader(dataset=train_dataset, batch_sampler=batch_sampler,
                                  num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=8, pin_memory=True)#, worker_init_fn=worker_init_fn)
    
    model.train()
    
    loss_fn = losses.cross_entropy_loss()
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path+'/log')

    iter_num = args.global_step
    lr_ = base_lr
    model.train()

    
    best_AUROC_avg, best_AUROC_avg2, best_AUROC_avg3 = 0.0, 0.0, 0.0
    best_Accus_avg, best_Accus_avg2, best_Accus_avg3 = 0.0, 0.0, 0.0
    best_Senss_avg, best_Senss_avg2, best_Senss_avg3 = 0.0, 0.0, 0.0
    best_Specs_avg, best_Specs_avg2, best_Specs_avg3 = 0.0, 0.0, 0.0
    best_F1_avg, best_F1_avg2, best_F1_avg3 = 0.0, 0.0, 0.0
    
    
    ####This part is used to initialize for a better clustering center 
    from collections import defaultdict
    center_init = defaultdict(int)
    each_class_num = defaultdict(int)
    #device_init = "cuda:" + str(int(args.gpu)+1)
    model_init =  create_model()
    
    for i, (_,_, ((image_batch, ema_image_batch),(image_batch2,ema_image_batch2)), label_batch) in enumerate(train_dataloader):
        image_batch, ema_image_batch, _, _, label_batch = image_batch.to("cuda:"+args.gpu), ema_image_batch.to("cuda:"+args.gpu), image_batch2.to("cuda:"+args.gpu), ema_image_batch2.to("cuda:"+args.gpu), label_batch.to("cuda:"+args.gpu)

        ema_inputs = ema_image_batch 
        inputs = image_batch 

        activations, outputs = model_init(inputs)
        #loss_init = loss_fn(outputs[:labeled_bs], label_batch[:labeled_bs])
        
        label = torch.argmax(label_batch[:labeled_bs],dim = 1).view(-1) #batch_size * 1
        for j in range(labeled_bs):
            
            center_init[int(label[j])] += activations[j].data
            each_class_num[int(label[j])] += 1
        
        for cls in each_class_num:
            center_init[cls] /= each_class_num[cls]
            center_init[cls].to("cuda:"+args.gpu)
    

    ###########################################################
    '''
    for i in range(7):
        print(center_init[i])
    '''
    #train
    for epoch in range(args.start_epoch, args.epochs):
    
        ##########日志记录显示
        meters_loss = MetricLogger(delimiter="  ")
        meters_loss_classification = MetricLogger(delimiter="  ")
        meters_loss_consistency = MetricLogger(delimiter="  ")
        meters_loss_consistency_relation = MetricLogger(delimiter="  ")
        meters_loss_consistency_cluster = MetricLogger(delimiter="  ")
        time1 = time.time()
        
        iter_max = len(train_dataloader)  # number of batches
        
        

        for i, (_,_, ((image_batch, ema_image_batch),(image_batch2,ema_image_batch2)), label_batch) in enumerate(train_dataloader):#items, index, image, torch.FloatTensor(label)
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            if i%20 == 0:
                print(i)
            #######Chuankai Change#############
            image_batch, ema_image_batch, image_batch2, ema_image_batch2, label_batch = image_batch.to("cuda:"+args.gpu), ema_image_batch.to("cuda:"+args.gpu), image_batch2.to("cuda:"+args.gpu), ema_image_batch2.to("cuda:"+args.gpu), label_batch.to("cuda:"+args.gpu)
            # unlabeled_image_batch = ema_image_batch[labeled_bs:]
            
            noise1 = torch.clamp(torch.randn_like(image_batch) * 0.1, -0.1, 0.1)
            noise2 = torch.clamp(torch.randn_like(ema_image_batch) * 0.1, -0.1, 0.1)

            noise3 = torch.clamp(torch.randn_like(image_batch2) * 0.1, -0.1, 0.1)
            noise4 = torch.clamp(torch.randn_like(ema_image_batch2) * 0.1, -0.1, 0.1)

            ema_inputs = ema_image_batch
            inputs = image_batch 

            ema_inputs2 = ema_image_batch2 #+ noise4
            inputs2 = image_batch2 #+ noise 3

            #print('inputs shape:',inputs.shape,'ema_inputs shape', ema_inputs.shape) batch_size x 3 x 224 x 224 =48 x 3 x 224 x 224
            activations, outputs = model(inputs)
            activations2, outputs2 = model(inputs2) # batch_size x classes
        
            with torch.no_grad():#已经detach了还有必要再no_grad()嘛
                ema_activations, ema_output = ema_model(ema_inputs)

                #########Chuankai Change
                ema_activations2,ema_output2 = ema_model(ema_inputs2)

            ## calculate the loss
            loss_classification = loss_fn(outputs[:labeled_bs], label_batch[:labeled_bs])
            #print(label_batch.shape,label_batch)
            ## stream2 loss
            loss_classification2 = loss_fn(outputs2[:labeled_bs], label_batch[:labeled_bs])
            loss = loss_classification + loss_classification2
            

            ######calculate the initial center based on labelled data 
            label = torch.argmax(label_batch[:labeled_bs],dim = 1).view(-1) #batch_size * 1
                
            ## MT loss (have no effect in the beginneing)
            if args.ema_consistency == 1:
                consistency_weight = get_current_consistency_weight(epoch) #Float 1e-5-1e-3

                ####Chuankai Changed###################################
                original_consistency_dist = torch.sum(losses.softmax_mse_loss(outputs, ema_output)) / batch_size
                consistency_dist = torch.sum(losses.softmax_mse_loss(outputs2, ema_output)) / batch_size #/ dataset.N_CLASSES
                consistency_dist2 = torch.sum(losses.softmax_mse_loss(outputs, ema_output2)) / batch_size  # float
                
                consistency_loss = consistency_weight * (consistency_dist + consistency_dist2) #+ original_consistency_dist)              

                # consistency_relation_dist = torch.sum(losses.relation_mse_loss_cam(activations, ema_activations, model, label_batch)) / batch_size
                consistency_relation_dist = torch.sum(losses.relation_mse_loss(activations, ema_activations2)) / batch_size #48*1024
                consistency_relation_dist2 = torch.sum(losses.relation_mse_loss(activations2, ema_activations)) / batch_size 
                consistency_relation_loss = consistency_weight * consistency_relation_dist  * args.consistency_relation_weight                
                
                
                consistency_cluster_relation_loss1 = torch.sum(losses.cluster_consistency_loss(activations2, ema_activations, outputs, ema_output, center_init, 7)) / batch_size
                consistency_cluster_relation_loss2 = torch.sum(losses.cluster_consistency_loss(activations, ema_activations2, outputs2, ema_output2, center_init, 7)) / batch_size
                
                consistency_cluster_relation_loss = consistency_weight * (consistency_cluster_relation_loss1 + consistency_cluster_relation_loss2)
                
                #############################################################
            else:
                consistency_loss = 0.0
                consistency_relation_loss = 0.0
                consistency_cluster_relation_loss = 0.0
                consistency_weight = 0.0
                consistency_dist = 0.0
             #+ consistency_loss

            if (epoch > 20) and (args.ema_consistency == 1):
                loss = loss_classification + consistency_loss + consistency_relation_loss + consistency_cluster_relation_loss

    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            # outputs_soft = F.softmax(outputs, dim=1)
            meters_loss.update(loss=loss)
            meters_loss_classification.update(loss=loss_classification)
            meters_loss_consistency.update(loss=consistency_loss)
            meters_loss_consistency_relation.update(loss=consistency_relation_loss)
            meters_loss_consistency_cluster.update(loss=consistency_cluster_relation_loss)
            iter_num = iter_num + 1

            # write tensorboard
            if i % 100 == 0:
                writer.add_scalar('lr', lr_, iter_num)
                writer.add_scalar('loss/loss', loss, iter_num)
                writer.add_scalar('loss/loss_classification', loss_classification, iter_num)
                writer.add_scalar('train/consistency_loss', consistency_loss, iter_num)
                writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
                writer.add_scalar('train/consistency_dist', consistency_dist, iter_num)
                writer.add_scalar('train/consistency_cluster_loss', consistency_cluster_relation_loss, iter_num)
                logging.info("\nEpoch: {}, iteration: {}/{}, ==> train <===, loss: {:.6f}, classification loss: {:.6f}, consistency loss: {:.6f}, consistency relation loss: {:.6f}, consistency_cluster_relation_loss: {:.6f}, consistency weight: {:.6f}, lr: {}"
                            .format(epoch, i, iter_max, meters_loss.loss.avg, meters_loss_classification.loss.avg, meters_loss_consistency.loss.avg, meters_loss_consistency_relation.loss.avg, meters_loss_consistency_cluster.loss.avg, consistency_weight, optimizer.param_groups[0]['lr']))

                image = inputs[-1, :, :]
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('raw/Image', grid_image, iter_num)

                image = ema_inputs[-1, :, :]
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('noise/Image', grid_image, iter_num)
        
        
        

        timestamp = get_timestamp()

        # validate student
        # 

        AUROCs, Accus, Senss, Specs = epochVal_metrics(model, val_dataloader)  
        AUROC_avg = np.array(AUROCs).mean()
        Accus_avg = np.array(Accus).mean()
        Senss_avg = np.array(Senss).mean()
        Specs_avg = np.array(Specs).mean()

        logging.info("\nVAL Student: Epoch: {}, iteration: {}".format(epoch, i))
        logging.info("\nVAL AUROC: {:6f}, VAL Accus: {:6f}, VAL Senss: {:6f}, VAL Specs: {:6f}"
                    .format(AUROC_avg, Accus_avg, Senss_avg, Specs_avg))
        logging.info("AUROCs: " + " ".join(["{}:{:.6f}".format(dataset.CLASS_NAMES[i], v) for i,v in enumerate(AUROCs)]))
        
        # test student
        # 
        AUROCs, Accus, Senss, Specs, Pre, F1 = epochVal_metrics_test(model, test_dataloader)
        
        AUROC_avg = np.array(AUROCs).mean()
        Accus_avg = np.array(Accus).mean()
        Senss_avg = np.array(Senss).mean()
        Specs_avg = np.array(Specs).mean()
        F1_avg = np.array(F1).mean()
        
        logging.info("\nTEST Student: Epoch: {}, iteration: {}".format(epoch, i))
        logging.info("\nTEST AUROC: {:6f}, TEST Accus: {:6f}, TEST Senss: {:6f}, TEST Specs: {:6f}, TEST F1: {:6f}"
                    .format(AUROC_avg, Accus_avg, Senss_avg, Specs_avg, F1_avg))
        logging.info("AUROCs: " + " ".join(["{}:{:.6f}".format(dataset.CLASS_NAMES[i], v) for i,v in enumerate(AUROCs)]))
 
        if AUROC_avg > best_AUROC_avg:
            best_F1_avg = F1_avg
            best_AUROC_avg = AUROC_avg
            best_Accus_avg = Accus_avg
            best_Senss_avg = Senss_avg
            best_Specs_avg = Specs_avg
            epoch_auc = epoch
        
        if Accus_avg > best_Accus_avg2:
            best_F1_avg2 = F1_avg
            best_AUROC_avg2 = AUROC_avg
            best_Accus_avg2 = Accus_avg
            best_Senss_avg2 = Senss_avg
            best_Specs_avg2 = Specs_avg
            epoch_acc = epoch

        if F1_avg > best_F1_avg3:
            best_F1_avg3 = F1_avg
            best_AUROC_avg3 = AUROC_avg
            best_Accus_avg3 = Accus_avg
            best_Senss_avg3 = Senss_avg
            best_Specs_avg3 = Specs_avg
            epoch_f1 = epoch

        # save model
        save_mode_path = os.path.join(snapshot_path + 'checkpoint/', 'epoch_' + str(epoch+1) + '.pth')
        torch.save({    'epoch': epoch + 1,
                        'global_step': iter_num,
                        'state_dict': model.state_dict(),
                        'ema_state_dict': ema_model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'epochs'    : epoch,
                        # 'AUROC'     : AUROC_best,
                   }
                   , save_mode_path
        )
        logging.info("save model to {}".format(save_mode_path))

        # update learning rate
        lr_ = lr_ * 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_
    logging.info("\nBest TEST Student:")
    logging.info("\nAppear in Epoch: {}, TEST AUROC: {:6f}, TEST Accus: {:6f}, TEST Senss: {:6f}, TEST Specs: {:6f}, TEST F1: {:6f}"
                    .format(epoch_auc, best_AUROC_avg, best_Accus_avg, best_Senss_avg, best_Specs_avg, best_F1_avg))
    logging.info("\nAppear in Epoch: {}, TEST AUROC: {:6f}, TEST Accus: {:6f}, TEST Senss: {:6f}, TEST Specs: {:6f}, TEST F1: {:6f}"
                    .format(epoch_acc, best_AUROC_avg2, best_Accus_avg2, best_Senss_avg2, best_Specs_avg2, best_F1_avg2))
    logging.info("\nAppear in Epoch: {}, TEST AUROC: {:6f}, TEST Accus: {:6f}, TEST Senss: {:6f}, TEST Specs: {:6f}, TEST F1: {:6f}"
                    .format(epoch_f1, best_AUROC_avg3, best_Accus_avg3, best_Senss_avg3, best_Specs_avg3, best_F1_avg3))
    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(iter_num+1)+'.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
