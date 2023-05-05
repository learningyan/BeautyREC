# encoding = utf-8

import os
import pdb
import time
import numpy as np

import torch
from torch import optim
from torch.autograd import Variable

import makeuploader.dataloaders as dl
from network import get_model

from options import opt

from utils import init_log, load_meta, save_meta
from mscv.summary import create_summary_writer, write_meters_loss,write_image

import misc_utils as utils
from mscv.image import tensor2im

# 初始化

# import pdb
# pdb.set_trace()

with torch.no_grad():
    # 初始化路径
    save_root = os.path.join(opt.checkpoint_dir, opt.tag)
    log_root = os.path.join(opt.log_dir, opt.tag)

    utils.try_make_dir(save_root)
    utils.try_make_dir(log_root)

    # Dataloader
    train_loader = dl.traindataloader
    test_loader = dl.testdataloader
    # 初始化日志
    logger = init_log(training=True)

    # 初始化训练的meta信息
    meta = load_meta(new=True)
    save_meta(meta)

    # 初始化模型
    Model = get_model(opt.model)
    model = Model(opt)

    # 暂时还不支持多GPU
    # if len(opt.gpu_ids):
    #     model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
    model = model.to(device=opt.device)

    # 加载预训练模型，恢复中断的训练
    if opt.load:
        load_epoch = model.load(opt.load)
        start_epoch = load_epoch + 1 if opt.resume else 1
    else:
        start_epoch = 1

    # 开始训练
    model.train()

    # 计算开始和总共的step
    print('Start training...')
    start_step = (start_epoch - 1) * len(train_loader)
    global_step = start_step
    total_steps = opt.epochs * len(train_loader)
    start = time.time()

    # Tensorboard初始化
    writer = create_summary_writer(log_root)

    start_time = time.time()





try:
    # 训练循环
    for epoch in range(start_epoch, opt.epochs + 1):
        for iteration, sample in enumerate(train_loader):
            global_step += 1
            # 计算剩余时间
            rate = (global_step - start_step) / (time.time() - start)
            remaining = (total_steps - global_step) / rate

            # 更新模型参数
            if opt.debug:
                output = model.update(sample)['output']
            else:
                try:
                    output = model.update(sample)['output']
                except:
                    continue
            # 获取当前学习率
            lr = model.get_lr()
            lr = lr if lr is not None else opt.lr

            # 显示进度条
            msg = f'lr:{round(lr, 6) : .6f} (loss) {str(model.avg_meters)} ETA: {utils.format_time(remaining)}'
            utils.progress_bar(iteration, len(train_loader), 'Epoch:%d' % epoch, msg)

            # # 训练时每100个step记录一下

            if global_step % 50 == 0:
                # pdb.set_trace()
                no_image = sample['nonmakeup_img'][0].detach().cpu().numpy().transpose([1,2,0])/2+0.5
                no_image = (no_image.copy()*255).astype(np.uint8)

                image = sample['makeup_img'][0].detach().cpu().numpy().transpose([1,2,0])/2+0.5
                image = (image.copy()*255).astype(np.uint8)
   
                
                output = output[0].detach().cpu().numpy().transpose([1,2,0])/2+0.5
                output = (output.copy()*255).astype(np.uint8)

                write_image(writer, 'train', 'no_makeupimage', no_image, global_step, 'HWC')
                write_image(writer, 'train', 'makeup_image', image, global_step, 'HWC')
                write_image(writer, 'train', 'output', output, global_step, 'HWC')

        # 每个epoch结束后的显示信息
        logger.info(f'Train epoch: {epoch}, lr: {round(lr, 6) : .6f}, (loss) ' + str(model.avg_meters))

        if epoch % opt.save_freq == 0 or epoch == opt.epochs:  # 最后一个epoch要保存一下
            model.save(epoch)



        # 训练中验证
        if epoch % opt.eval_freq == 0:
            model.eval()
            
            for j, sample in enumerate(test_loader):
                if j%5 ==0:
                    try:
                        pred = model(sample)
                    except:
                        continue
                    no_image = sample['nonmakeup_img'][0].detach().cpu().numpy().transpose([1,2,0])/2+0.5
                    no_image = (no_image.copy()*255).astype(np.uint8)

                    image = sample['makeup_img'][0].detach().cpu().numpy().transpose([1,2,0])/2+0.5
                    image = (image.copy()*255).astype(np.uint8)

                    pred = pred[0].detach().cpu().numpy().transpose([1,2,0])/2+0.5
                    pred = (pred.copy()*255).astype(np.uint8)

                    write_image(writer, f'test{j}', 'no_makeupimage', no_image, epoch, 'HWC')
                    write_image(writer, f'test{j}', 'makeup_image', image, epoch, 'HWC')
                    write_image(writer, f'test{j}', 'output', pred, epoch, 'HWC')

            # break

            model.train()

        model.step_scheduler()

    meta = load_meta()
    meta[-1]['finishtime'] = utils.get_time_stamp()
    save_meta(meta)

except Exception as e:

    if opt.tag != 'cache':
        with open('run_log.txt', 'a') as f:
            f.writelines('    Error: ' + str(e)[:120] + '\n')

    meta = load_meta()
    meta[-1]['finishtime'] = utils.get_time_stamp()
    save_meta(meta)

    raise Exception('Error')  # 再引起一个异常，这样才能打印之前的错误信息

except:  # 其他异常，如键盘中断等
    meta = load_meta()
    meta[-1]['finishtime'] = utils.get_time_stamp()
    save_meta(meta)