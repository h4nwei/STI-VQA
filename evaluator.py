from pathlib import Path
import torch

from utility import AverageMeter, IQAPerformance


class Evaluator:
    def __init__(self, args, model, loader):
        self.args = args
        self.log_dir = Path(args.log_dir)
        self.loader_test = loader
        self.model = torch.nn.DataParallel(model).cuda()
        self.max_len = args.max_len
 
    def predict(self):
        self.model.eval()
        pred_list = []
        mos_list = []
        perf = IQAPerformance(self.log_dir)
        with torch.no_grad():
            for i, (x, length, y, scale) in enumerate(self.loader_test, start=1):
                x, y = self.prepare(x, y)
                length = length.cuda(non_blocking=True)
                scale = scale.cuda(non_blocking=True)
                _,_,_,y_pred = self.model(x, length, self.max_len)
                # y_pred = self.model(x, length)
                y_pred = y_pred.squeeze()
                y_pred = y_pred * scale
                
                y = y * scale   

                perf.update(y_pred, y)
                pred_list.extend([p.item() for p in y_pred])
                mos_list.extend([s.item() for s in y])
        
        corr = perf.compute(is_plot=True, fig_name=f'validation.png')
        with open(self.log_dir / 'Testing.csv', 'w') as f:
            for mos, pred in zip(mos_list, pred_list):
                f.write(f"{float(mos):.3f}, {float(pred):.3f}\n")
        print(f"Testing Result:\nSRCC {corr['srcc']:.4f} | KRCC {corr['krcc']:.4f} | PLCC {corr['plcc']:.4f} | RMSE {corr['rmse']:.4f}")
        return corr

    def prepare(self, x, y):
        if type(x) in [list, tuple]:
            x = [x_.cuda(non_blocking=True) for x_ in x]
        else:
            x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        return [x, y]
