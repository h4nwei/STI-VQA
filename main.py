import numpy as np
import random
from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader
import os
from params import args
from dataset import  DeepFeatDataset
from loss import Loss 
from train import Trainer
from evaluator import Evaluator

os.environ['CUDA_VISIBLE_DEVICES']='4'

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(args.seed)
random.seed(args.seed)

def main():
    if not args.test_only:
        with open(os.path.join(args.log_dir, 'args.txt'), 'w') as f:
            json_dict = {k:str(v) for k, v in args.__dict__.items()}
            json.dump(json_dict, f, indent=4)

        if args.model == 'sti-vqa':
            from network.sti_vqa import STI_VQA
            ds_train = DeepFeatDataset(args, phase='train')
            ds_val = DeepFeatDataset(args, phase='test')
            model = STI_VQA(input_dim=args.d_feat,
                    mlp_dim=args.mlp_dim, 
                    dim_head=args.dim_head,
                    output_channel=args.output_channel, 
                    depth=args.depth, 
                    heads=args.att_head,
                    pool = args.pool, 
                    dropout = args.emb_dropout,
                    emb_dropout = args.emb_dropout,
                    max_length=args.max_len)

        
        else:
            raise NotImplementedError

        loader = {}
        loader['train'] = DataLoader(dataset=ds_train, batch_size=args.batch_size, shuffle=True,
                pin_memory=True, num_workers=args.n_threads)
        loader['val'] = DataLoader(dataset=ds_val, batch_size=args.batch_size, shuffle=False,
                pin_memory=True, num_workers=args.n_threads)
        loss = Loss(args.loss)
        t = Trainer(args, model, loader, loss)
        _best_info = t.main_worker()

        print('\nBest info:\n')
        print(_best_info)

        args.pre_train = os.path.join(args.ckpt_dir, 'best_val') + '.pth'
    
    # Predict on testing set
    if not args.pre_train:
        raise ValueError("A pre-trained model is needed!")

    if args.model == 'vit':
        from network.sti_vqa import STI_VQA
        ds_test = DeepFeatDataset(args, phase='test')
        model = STI_VQA(input_dim=args.d_feat,
                    mlp_dim=args.mlp_dim, 
                    dim_head=args.dim_head,
                    output_channel=args.output_channel, 
                    depth=args.depth, 
                    heads=args.att_head,
                    pool = args.pool, 
                    dropout = args.emb_dropout,
                    emb_dropout = args.emb_dropout,
                    max_length=args.max_len)
    
    else:
        raise NotImplementedError

    model.load_state_dict(torch.load(args.pre_train, map_location=lambda storage, loc: storage), strict=False)
    loader = DataLoader(dataset=ds_test, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=0)
    t = Evaluator(args, model, loader)
    return t.predict()


if __name__ == '__main__':
    result_dict = {}
    data_info_path_list = sorted(os.listdir(args.data_info_path))
    srocc_list = np.zeros(len(data_info_path_list))
    krocc_list = np.zeros(len(data_info_path_list))
    plcc_list = np.zeros(len(data_info_path_list))
    rmse_list = np.zeros(len(data_info_path_list))
    for i, data_info_path in enumerate(data_info_path_list):
        args.data_info_dir = os.path.join(args.data_info_path, data_info_path)
        args.data_info_split_idx = str(i)
        args.log_dir = os.path.join(args.log_root, args.model_name, args.dataset_name, str(i))
        args.ckpt_dir = os.path.join(args.ckpt_root, args.model_name, args.dataset_name, str(i))
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)
        test_info = main()
        result_dict[i] = {'test_info': test_info}
        srocc_list[i] = test_info['srcc']
        krocc_list[i] = test_info['krcc']
        plcc_list[i] = test_info['plcc']
        rmse_list[i] = test_info['rmse']

    print('mean srocc: {}[{}]({})'.format(np.mean(srocc_list), np.median(srocc_list), np.std(srocc_list)))
    print('mean krocc: {}[{}]({})'.format(np.mean(krocc_list), np.median(krocc_list), np.std(krocc_list)))
    print('mean plcc: {}[{}]({})'.format(np.mean(plcc_list), np.median(plcc_list), np.std(plcc_list)))
    print('mean rmse: {}[{}]({})'.format(np.mean(rmse_list), np.median(rmse_list), np.std(rmse_list)))
    result_dict['overall'] = {'SROCC': {'mean': np.mean(srocc_list), 'median': np.median(srocc_list), 'std': np.std(srocc_list)},
                              'KROCC': {'mean': np.mean(krocc_list), 'median': np.median(krocc_list), 'std': np.std(krocc_list)},
                              'PLCC': {'mean': np.mean(plcc_list), 'median': np.median(plcc_list), 'std': np.std(plcc_list)},
                              'RMSE': {'mean': np.mean(rmse_list), 'median': np.median(rmse_list), 'std': np.std(rmse_list)},}

    # result_folder = os.path.join(args.exp_settings['result_folder'], opt.model_info['model'],
    #                              str(opt.exp_settings['exp_id']), opt.dataset_info['dataset_name'])
    with open(os.path.join(args.ckpt_dir, 'results.json'), 'w') as f:
        json.dump(result_dict, f)
