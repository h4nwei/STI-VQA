from pathlib import Path
import numpy as np
from scipy import stats
import time
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr, kendalltau
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class AverageMeter(object):
    def __init__(self):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.round(self.sum / self.count, 4)


class IQAPerformance(object):
    """
    Evaluation of IQA methods using SROCC, KROCC, PLCC, RMSE, MAE, OR.
    `update` must receive output of the form (y_pred, y).
    """
    def __init__(self, save_dir='.'):
        self.pred = []
        self.mos = []
        self.save_dir = Path(save_dir)

    def reset(self):
        self.pred = []
        self.mos = []

    def update(self, pred, mos):
        if mos.shape[0] == 1:
            pred = [pred.item()]
            mos = [mos.item()]
        else:
            pred = pred.tolist()
            mos = mos.tolist()

        self.pred.extend(pred)
        self.mos.extend(mos)

    def compute(self, is_plot=False, fig_name='scatter_plot.png'):
        corr = {}
        mos = np.reshape(np.asarray(self.mos), (-1,))
        pred = np.reshape(np.asarray(self.pred), (-1,))
        plcc, srcc, krcc, rmse = correlation_evaluation(pred, mos, is_plot=is_plot, plot_path=self.save_dir/fig_name)
        corr['plcc'], corr['srcc'], corr['krcc'], corr['rmse'] = np.round((plcc, srcc, krcc, rmse), 4)
        return corr


def logistic(X, beta1, beta2, beta3, beta4, beta5):
    logistic_part = 0.5 - 1./(1 + np.exp(beta2 * (X - beta3)))
    yhat = beta1 * logistic_part + beta4 * X + beta5
    return yhat


def correlation_evaluation(obj_score, mos, is_plot, plot_path,
    xlabel='predicted score', ylabel='mean opinion score'):
    r""" correlation evaluation between MOS and objective scores
    after nonlinear regression [fitting_5parameter]

    Args:
        obj_score:  multi-gpu checkpoint dictionary
        mos:        the ground-truth mean opinion score
    """
    beta1 = np.max(mos)
    beta2 = np.min(mos)
    beta3 = np.mean(obj_score)
    beta = [beta1, beta2, beta3, 0.1, 0.1]  # inital guess for non-linear fitting

    obj_score = np.array(obj_score)
    mos = np.array(mos)
    fit_stat = ''
    try:
        popt, _ = curve_fit(logistic, xdata=obj_score, ydata=mos, p0=beta, maxfev=10000)
    except:
        popt = beta
        fit_stat = '[nonlinear reg failed]'
    ypred = logistic(obj_score, popt[0], popt[1], popt[2], popt[3], popt[4])

    plcc, _ = pearsonr(mos, ypred)
    #srcc, _ = spearmanr(mos, ypred)
    srcc, _ = spearmanr(mos, obj_score)
    #krcc, _ = kendalltau(mos, ypred)
    krcc, _ = kendalltau(mos, obj_score)
    rmse = np.sqrt(np.mean( (ypred - mos) **2))
    if is_plot:
        _pseu_x = np.linspace(np.min(obj_score), np.max(obj_score), 100)
        _pseu_pred = logistic(_pseu_x, popt[0], popt[1], popt[2], popt[3], popt[4])
        plt.style.use('ggplot')

        fig = plt.figure()
        plt.plot(obj_score, mos, marker='2', color=mcolors.CSS4_COLORS['darkcyan'], linestyle='')
        plt.plot(_pseu_x, _pseu_pred, color=mcolors.CSS4_COLORS['firebrick'], linestyle='-')
        for offset in [+rmse, -rmse]:
            plt.plot(_pseu_x, _pseu_pred+offset,
                    color=mcolors.CSS4_COLORS['orangered'], linestyle='--', linewidth=1.0)

        plt.title('scatter plot {} \n plcc: {:0.3f}, srcc: {:0.3f}, rmse: {:0.3f}'.format(fit_stat,
            plcc, srcc, rmse))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        fig.savefig(plot_path, dpi=400)
        plt.close()
    return float(plcc), float(srcc), float(krcc), float(rmse)
