import time
from PIL import Image
from numpy.core.fromnumeric import squeeze
import torch
from torch.utils.tensorboard import SummaryWriter
import os
from optimizer import make_optimizer
from utility import AverageMeter, IQAPerformance


class Trainer:
    def __init__(self, args, model, loader, loss):
        self.args = args
        self.log_dir = args.log_dir
        self.writer = SummaryWriter(self.log_dir)

        self.loader_train, self.loader_test = loader['train'], loader['val'] #validation
        self.model = torch.nn.DataParallel(model).cuda()
        self.loss = loss
        self.optimizer = make_optimizer(args, self.model)

        self.cur_epoch = 0
        self.cur_step = 0
        self.bestinfo = {'epoch': -1, 'loss': 1e6, 'srcc': 0.0, 'plcc': 0.0, 'krcc': 0.0}
        self.epoch_step = int(len(self.loader_train))
        self.best_perf = -0.01
        self.data_info_split_idx = args.data_info_split_idx
        self.ckpt_dir = args.ckpt_dir
        self.max_len = args.max_len
        # train_model_name = './checkpoints/train.pth'
        # self.train_model_name = os.path.join(args.ckpt_dir, 'train') + '_{}.pth'.format(args.data_info_split_idx)
        # best_val_model_name = './checkpoints/best_val.pth'
        # self.best_val_model_name = os.path.join(args.ckpt_dir, 'best_val') + '_{}.pth'.format(args.data_info_split_idx)
    

    def main_worker(self):        
        for epoch in range(1, self.args.epochs+1):
            epoch_start = time.time()
            self.cur_epoch = epoch            
            is_plot = epoch % self.args.save_scatter == 0

            train_loss, train_corr = self.train(is_plot=is_plot)
            self.optimizer.schedule()

            self.writer.add_scalar('learning rate', self.optimizer.get_lr(), epoch)
            for k, v in train_loss.items():
                self.writer.add_scalars(k, {'train': v}, epoch)
            self.writer.add_scalars('plcc', {'train': train_corr['plcc']}, epoch)
            self.writer.add_scalars('srcc', {'train': train_corr['srcc']}, epoch)
            self.writer.add_scalars('krcc', {'train': train_corr['krcc']}, epoch)

            if epoch >= 100 and epoch % self.args.save_weights == 0:
                torch.save(self.model.module.state_dict(), self.ckpt_dir/f'checkoutpoint_ep{epoch:03d}.pth.tar')

            if epoch % self.args.test_every == 0:
                test_loss, test_corr = self.test(is_plot=is_plot)

                is_best = test_corr['srcc'] > self.bestinfo['srcc']
                if is_best:
                    self.bestinfo['epoch'] = epoch
                    self.bestinfo['loss'] = test_loss['Total']
                    self.bestinfo['srcc'] = test_corr['srcc']
                    self.bestinfo['plcc'] = test_corr['plcc']
                    self.bestinfo['krcc'] = test_corr['krcc']
                    torch.save(self.model.module.state_dict(), os.path.join(self.ckpt_dir, 'best_val') + '.pth')

                for k, v in test_loss.items():
                    self.writer.add_scalars(k, {'test': v}, epoch)
                self.writer.add_scalars('plcc', {'test': test_corr['plcc']}, epoch)
                self.writer.add_scalars('srcc', {'test': test_corr['srcc']}, epoch)
                self.writer.add_scalars('krcc', {'test': test_corr['krcc']}, epoch)

            tr_time = time.time() - epoch_start
            print(f"#Ep -> {epoch}/{self.args.epochs} | Time -> {tr_time:.1f}s | Best -> Ep {self.bestinfo['epoch']}, Loss {self.bestinfo['loss']:.4f}, SRCC {self.bestinfo['srcc']:.4f}")
        return self.bestinfo

    def train(self, is_plot=False):
        time_data = AverageMeter() 
        time_model = AverageMeter()

        train_loss = {}
        
        for l in self.loss.loss:
            train_loss[l['type']] = AverageMeter()
        train_perf = IQAPerformance(self.log_dir)

        self.model.train()
        time_start = time.time()
        print('Adam learning rate: {:.8f}'.format(self.optimizer.param_groups[0]['lr']))
        for bi, (x, length, y, scale) in enumerate(self.loader_train, start=1):
            x, y = self.prepare(x, y)
            length = length.cuda(non_blocking=True)
            scale = scale.cuda(non_blocking=True)
            time_mid = time.time()
            time_data.update(time_mid - time_start)

            y_pred_0, y_pred_1, y_pred_2, y_pred_3  = self.model(x, length, self.max_len)
            # y_pred = self.model(x, length)
            # y_pred = y_pred
        
            loss_0, loss_items_0 = self.loss(input_mos=y_pred_0.float(), target_mos=y.unsqueeze(1).float())
            loss_1, loss_items_1 = self.loss(input_mos=y_pred_1.float(), target_mos=y.unsqueeze(1).float())
            loss_2, loss_items_2 = self.loss(input_mos=y_pred_2.float(), target_mos=y.unsqueeze(1).float())
            loss_3, loss_items = self.loss(input_mos=y_pred_3.float(), target_mos=y.unsqueeze(1).float())
            loss = loss_0 + loss_1 + loss_2 + loss_3
            # y_pred = self.model(x, length)
            
            self.optimizer.zero_grad()
            # loss, loss_items = self.loss(input_mos=y_pred.unsqueeze(1), target_mos=y.unsqueeze(1))
            # loss, loss_items = self.loss(input_mos=y_pred, target_mos=y.unsqueeze(1))
            loss.backward()
            self.optimizer.step()
            y_pred_3 = y_pred_3.squeeze() * scale
            y = y * scale

            for k, v in loss_items.items():
                train_loss[k].update(v.item(), y.size(0))
            train_perf.update(y_pred_3, y)

            time_model.update(time.time() - time_mid)
            time_start = time.time()

        for k, l in train_loss.items():
            train_loss[k] = l.avg
        train_corr = train_perf.compute(is_plot=is_plot, fig_name=f'train_{self.cur_epoch}.png')

        print(f'Train Time -- Data {time_data.avg * 1000:.1f}ms | Model {time_model.avg * 1000:.1f}ms')
        print('      Loss -- ' + ' | '.join(f'{k} {v:.4f}' for k, v in train_loss.items()))
        print(f"      Corr -- SRCC {train_corr['srcc']:.4f} | KRCC {train_corr['krcc']:.4f} | PLCC {train_corr['plcc']:.4f} | RMSE {train_corr['rmse']:.4f}")
        return train_loss, train_corr

    def test(self, is_plot=False):
        test_loss = {}
        for l in self.loss.loss:
            test_loss[l['type']] = AverageMeter()
        test_perf = IQAPerformance(self.log_dir)

        self.model.eval()
        with torch.no_grad():
            for bi, (x, length, y, scale) in enumerate(self.loader_test, start=1):
                x, y = self.prepare(x, y)
                length = length.cuda(non_blocking=True)
                scale = scale.cuda(non_blocking=True)
                y_pred_0, y_pred_1, y_pred_2, y_pred_3  = self.model(x, length, self.max_len)
                # y_pred = self.model(x, length)
                # y_pred = y_pred
           
                loss_0, loss_items_0 = self.loss(input_mos=y_pred_0, target_mos=y.unsqueeze(1))
                loss_1, loss_items_1 = self.loss(input_mos=y_pred_1, target_mos=y.unsqueeze(1))
                loss_2, loss_items_2 = self.loss(input_mos=y_pred_2, target_mos=y.unsqueeze(1))
                loss_3, loss_items = self.loss(input_mos=y_pred_3, target_mos=y.unsqueeze(1))
                loss = loss_0 + loss_1 + loss_2 + loss_3
                # loss, loss_items = self.loss(input_mos=y_pred.unsqueeze(1), target_mos=y.unsqueeze(1))
                y_pred_3 = y_pred_3.squeeze() * scale
                y = y * scale 

                for k, v in loss_items.items():
                    test_loss[k].update(v.item(), y.size(0))
                test_perf.update(y_pred_3, y)
    
        for k, l in test_loss.items():
            test_loss[k] = l.avg
        test_corr = test_perf.compute(is_plot=is_plot, fig_name=f'test_{self.cur_epoch}.png')

        print('Test  Loss -- ' + ' | '.join(f'{k} {v:.4f}' for k, v in test_loss.items()))
        print(f"      Corr -- SRCC {test_corr['srcc']:.4f} | KRCC {test_corr['krcc']:.4f} | PLCC {test_corr['plcc']:.4f} | RMSE {test_corr['rmse']:.4f}")
        return test_loss, test_corr

    def prepare(self, x, y):
        if type(x) in [list, tuple]:
            x = [x_.cuda(non_blocking=True) for x_ in x]
        else:
            x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        return [x, y]
