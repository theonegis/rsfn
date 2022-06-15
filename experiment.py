import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

from model import *
from data import *
from utils import *

from timeit import default_timer as timer
from datetime import datetime
import pandas as pd
import numpy as np
import shutil

from torchgan.losses import LeastSquaresDiscriminatorLoss, LeastSquaresGeneratorLoss


class Experiment(object):
    def __init__(self, option):
        self.device = torch.device('cuda' if option.cuda else 'cpu')
        self.scale = SCALE_FACTOR
        self.image_size = option.image_size

        self.save_dir = option.save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.train_dir = self.save_dir / 'train'
        self.train_dir.mkdir(exist_ok=True)
        self.test_dir = self.save_dir / 'test'
        self.test_dir.mkdir(exist_ok=True)
        self.history = self.train_dir / 'history.csv'
        self.best = self.train_dir / 'best.pth'
        self.last_g = self.train_dir / 'generator.pth'
        self.last_d = self.train_dir / 'discriminator.pth'

        self.logger = get_logger()
        self.logger.info('Model initialization')

        pretrained = 'assets/autoencoder.pth'
        self.pretrained = AutoEncoder().to(self.device)
        load_pretrained(self.pretrained, pretrained)
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        device_ids = [i for i in range(option.ngpu)]
        if option.cuda and option.ngpu > 1:
            self.generator = nn.DataParallel(self.generator, device_ids)
            self.discriminator = nn.DataParallel(self.discriminator, device_ids)

        self.criterion = ReconstructionLoss(self.pretrained)
        self.g_loss = LeastSquaresGeneratorLoss()
        self.d_loss = LeastSquaresDiscriminatorLoss()
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=option.lr)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=option.lr)

        self.logger.info(str(self.generator))
        self.logger.info(str(self.discriminator))

    def train_on_epoch(self, n_epoch, data_loader):
        self.generator.train()
        self.discriminator.train()
        epg_loss = AverageMeter()
        epd_loss = AverageMeter()
        epg_error = AverageMeter()

        batches = len(data_loader)
        self.logger.info(f'Epoch[{n_epoch}] - {datetime.now()}')
        for idx, data in enumerate(data_loader):
            t_start = timer()
            data = [im.to(self.device) for im in data]
            inputs, target = data[:-1], data[-1]
            prediction = self.generator(inputs)
            ############################
            # (1) Update D network
            ###########################
            self.discriminator.zero_grad()
            self.generator.zero_grad()
            d_loss = self.d_loss(self.discriminator(torch.cat((target, inputs[-1]), 1)),
                                 self.discriminator(torch.cat((prediction.detach(), inputs[-1]), 1)))
            d_loss.backward()
            self.d_optimizer.step()
            epd_loss.update(d_loss.item())
            ############################
            # (2) Update G network
            ###########################
            a_loss = (self.criterion(prediction, target) + 5e-3 *
                      self.g_loss(self.discriminator(torch.cat((prediction, inputs[-1]), 1))))
            a_loss.backward()
            self.g_optimizer.step()
            epg_loss.update(a_loss.item())
            mse = F.mse_loss(prediction.detach(), target).item()
            epg_error.update(mse)
            t_end = timer()
            self.logger.info(f'Epoch[{n_epoch} {idx}/{batches}] - '
                             f'A-Loss: {a_loss.item():.6f} - '
                             f'D-Loss: {d_loss.item():.6f} - '
                             f'MSE: {mse:.6f} - '
                             f'Time: {t_end - t_start}s')

        self.logger.info(f'Epoch[{n_epoch}] - {datetime.now()}')
        # 记录Checkpoint
        save_checkpoint(self.generator, self.g_optimizer, self.last_g)
        save_checkpoint(self.discriminator, self.d_optimizer, self.last_d)
        return epg_loss.avg, epd_loss.avg, epg_error.avg

    @torch.no_grad()
    def test_on_epoch(self, data_loader):
        self.generator.eval()
        self.discriminator.eval()
        epoch_error = AverageMeter()
        for data in data_loader:
            data = [im.to(self.device) for im in data]
            inputs, target = data[:-1], data[-1]
            prediction = F.relu(self.generator(inputs), True)
            error = F.mse_loss(prediction, target).item()
            epoch_error.update(error)
        return epoch_error.avg

    def train(self, train_dir, val_dir, patch_stride, batch_size,
              epochs=30, num_workers=0, resume=True):
        last_epoch = -1
        if resume and self.history.exists():
            df = pd.read_csv(self.history)
            last_epoch = int(df.iloc[-1]['epoch'])
            load_checkpoint(self.last_g, self.generator, optimizer=self.g_optimizer)
            load_checkpoint(self.last_d, self.discriminator, optimizer=self.d_optimizer)
        start_epoch = last_epoch + 1
        least_error = float('inf')

        # 加载数据
        self.logger.info('Loading data...')
        train_set = PatchSet(train_dir, self.image_size, PATCH_SIZE, patch_stride, mode=Mode.TRAINING)
        val_set = PatchSet(val_dir, self.image_size, PATCH_SIZE, patch_stride, mode=Mode.VALIDATION)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size,
                                num_workers=num_workers)

        self.logger.info('Training...')
        g_scheduler = ExponentialLR(self.g_optimizer, 0.99)
        d_scheduler = ExponentialLR(self.d_optimizer, 0.99)
        for epoch in range(start_epoch, epochs + start_epoch):
            self.logger.info(f"Learning rate for Generator: {self.g_optimizer.param_groups[0]['lr']}")
            self.logger.info(f"Learning rate for Discriminator: {self.d_optimizer.param_groups[0]['lr']}")
            train_g_loss, train_d_loss, train_g_error = self.train_on_epoch(epoch, train_loader)
            val_error = self.test_on_epoch(val_loader)
            csv_header = ['epoch', 'train_g_loss', 'train_d_loss', 'train_error', 'val_error']
            csv_values = [epoch, train_g_loss, train_d_loss, train_g_error, val_error]
            log_csv(self.history, csv_values, header=csv_header)
            g_scheduler.step()
            d_scheduler.step()

            if val_error < least_error:
                least_error = val_error
                shutil.copy(str(self.last_g), str(self.best))

    @torch.no_grad()
    def test(self, test_dir, patch_size, num_workers=0):
        load_checkpoint(self.best, self.generator)
        self.generator.eval()
        patch_size = make_tuple(patch_size)
        self.logger.info('Predicting...')
        # 记录测试文件夹中的文件路径，用于最后投影信息的匹配
        image_dirs = [p for p in test_dir.glob('*') if p.is_dir()]
        image_paths = [get_pair_path(d, Mode.PREDICTION) for d in image_dirs]

        # 在预测阶段，对图像进行切块的时候必须刚好裁切完全，这样才能在预测结束后进行完整的拼接
        assert self.image_size[0] % patch_size[0] == 0
        assert self.image_size[1] % patch_size[1] == 0
        rows = int(self.image_size[1] / patch_size[1])
        cols = int(self.image_size[0] / patch_size[0])
        n_blocks = rows * cols  # 一张图像中的分块数目
        test_set = PatchSet(test_dir, self.image_size, patch_size, mode=Mode.PREDICTION)
        test_loader = DataLoader(test_set, batch_size=1, num_workers=num_workers)

        scale_factor = 10000
        im_count = 0
        patches = []
        for data in test_loader:
            inputs = [im.to(self.device) for im in data]
            name = image_paths[im_count][-1].name
            if len(patches) == 0:
                t_start = timer()
                self.logger.info(f'Predict on image {name}')

            # 分块进行预测（每次进入深度网络的都是影像中的一块）
            prediction = F.relu(self.generator(inputs), True)
            prediction = prediction.squeeze_().cpu().numpy()
            prediction = (prediction * scale_factor).astype(np.int16)
            patches.append(prediction)

            # 完成一张影像以后进行拼接
            if len(patches) == n_blocks:
                result = np.empty((NUM_BANDS, *self.image_size), dtype=np.int16)
                block_count = 0
                for i in range(rows):
                    row_start = i * patch_size[1]
                    for j in range(cols):
                        col_start = j * patch_size[0]
                        result[:,
                        col_start: col_start + patch_size[0],
                        row_start: row_start + patch_size[1],
                        ] = patches[block_count]
                        block_count += 1
                patches.clear()
                # 存储预测影像结果
                prototype = str(image_paths[im_count][0])
                save_array_as_tif(result, self.test_dir / name, prototype=prototype)
                im_count += 1
                t_end = timer()
                self.logger.info(f'Time cost: {t_end - t_start}s')
