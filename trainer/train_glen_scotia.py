"""
    Trainer for Glen Scotia

    @author: Younghyun Kim
    Created on 2022.04.27
"""
import argparse
import logging
import os
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from kingsman.cfg.knights_configs import GLEN_SCOTIA_TUNE_CONFIG
from kingsman.dataset import InvestingStrategyGeneratorDataset
from kingsman.knights import GlenScotia

def train_glen_scotia(config: dict, checkpoint_dir=False):
    """
        Train Glen Scotia
    """
    if config['device'] == 'cuda':
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
    else:
        device = config['device']

    train_loader, valid_loader = get_data_loaders(
        features_path=config['features_path'],
        best_worst_pos_series_path=config['best_worst_pos_series_path'],
        best_st_series_path=config['best_st_series_path'],
        worst_st_series_path=config['worst_st_series_path'],
        best_rebal_series_path=config['best_rebal_series_path'],
        worst_rebal_series_path=config['worst_rebal_series_path'],
        valid_prob=config['valid_prob'],
        batch_size=config['batch_size'],
        valid_batch_size=config['valid_batch_size'],
        window=config['window'], eps=config['eps'])

    model = GlenScotia(config=config).to(device)

    if config['load_model']:
        model.load_state_dict(
            torch.load(config['load_model_path'],
                    map_location=device))
    model.eval()

    optimizer = optim.Adam(model.parameters(),
                            lr=config['lr'],
                            amsgrad=config['amsgrad'],
                            betas=(config['beta_1'], config['beta_2']))

    if checkpoint_dir:
        with open(os.path.join(checkpoint_dir, "checkpoint")) as f:
            model_state, optimizer_state = torch.load(f)

        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    for epoch in range(config['epoch_size']):
        train(model, optimizer, train_loader, device)

        mean_loss, init_loss, rebal_loss =\
            validation(model, valid_loader, device)

        if epoch % config['checkpoint_epoch'] == 0:
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(),
                            optimizer.state_dict()), path)
        tune.report(mean_loss=mean_loss,
                init_loss=init_loss, rebal_loss=rebal_loss)

def train(model, optimizer, train_loader, device=None):
    """
        train method
    """
    device = device or torch.device("cpu")

    model.train()

    for targets in train_loader:
        features_bw = targets['features_bw'].to(device)
        features_rebal = targets['features_rebal'].to(device)
        best_st = targets['best_st'].to(device)
        worst_st = targets['worst_st'].to(device)
        best_rebal_st = targets['best_rebal_st'].to(device)
        worst_rebal_st = targets['worst_rebal_st'].to(device)
        best_idx = targets['best_idx'].to(device)
        worst_idx = targets['worst_idx'].to(device)
        initial_idx = targets['initial_idx'].to(device)
        rebal_idx = targets['rebal_idx'].to(device)

        bw_idx = torch.cat((best_idx, worst_idx), dim=0)

        # Initial Investing
        features_bw = features_bw.repeat(2, 1, 1, 1)
        initial_st = torch.cat((best_st, worst_st), dim=0)
        initial_idx = initial_idx.repeat(2)

        init_preds = model(features_bw, initial_st,
                        bw_idx, initial_idx,
                        enc_time_mask=False,
                        dec_time_mask=True)

        init_loss = F.cross_entropy(
            init_preds[:, 0], initial_st.view(-1))
        optimizer.zero_grad()
        init_loss.backward()
        optimizer.step()

        # Rebalancing
        features_rebal = features_rebal.repeat(2, 1, 1, 1)
        rebal_st = torch.cat((best_rebal_st, worst_rebal_st), dim=0)
        rebal_idx = rebal_idx.repeat(2)

        rebal_preds = model(features_rebal, rebal_st,
                            bw_idx, rebal_idx,
                            enc_time_mask=False,
                            dec_time_mask=True)

        rebal_loss = F.cross_entropy(
            rebal_preds[:, 1], rebal_st[:, 1].view(-1))
        optimizer.zero_grad()
        rebal_loss.backward()
        optimizer.step()

def validation(model, data_loader, device=None):
    """
        validation method
    """
    device = device or torch.device("cpu")
    model.eval()

    mean_loss = 0.
    loss_i = 0.
    loss_r = 0.
    for batch_idx, targets in enumerate(data_loader):
        features_bw = targets['feathres_bw'].to(device)
        features_rebal = targets['features_rebal'].to(device)
        best_st = targets['best_st'].to(device)
        worst_st = targets['worst_st'].to(device)
        best_rebal_st = targets['best_rebal_st'].to(device)
        worst_rebal_st = targets['worst_rebal_st'].to(device)
        best_idx = targets['best_idx'].to(device)
        worst_idx = targets['worst_idx'].to(device)
        initial_idx = targets['initial_idx'].to(device)
        rebal_idx = targets['rebal_idx'].to(device)

        bw_idx = torch.cat((best_idx, worst_idx), dim=0)

        with torch.no_grad():
            # Initial Investing
            features_bw = features_bw.repeat(2, 1, 1, 1)
            initial_st = torch.cat((best_st, worst_st), dim=0)
            initial_idx = initial_idx.repeat(2)

            init_preds = model(features_bw, initial_st,
                            bw_idx, initial_idx,
                            enc_time_mask=False,
                            dec_time_mask=True)

            init_loss = F.cross_entropy(
                init_preds[:, 0], initial_st.view(-1))

            # Rebalancing
            features_rebal = features_rebal.repeat(2, 1, 1, 1)
            rebal_st = torch.cat((best_rebal_st, worst_rebal_st), dim=0)
            rebal_idx = rebal_idx.repeat(2)

            rebal_preds = model(features_rebal, rebal_st,
                                bw_idx, rebal_idx,
                                enc_time_mask=False,
                                dec_time_mask=True)

            rebal_loss = F.cross_entropy(
                rebal_preds[:, 1], rebal_st[:, 1].view(-1))

        mean_loss += (init_loss.item() + rebal_loss.item())
        loss_i += init_loss.item()
        loss_r += rebal_loss.item()

    mean_loss /= (batch_idx + 1)
    loss_i /= (batch_idx + 1)
    loss_r /= (batch_idx + 1)

    return mean_loss, loss_i, loss_r

def get_data_loaders(features_path, best_st_series_path,
                    worst_st_series_path,
                    best_worst_pos_series_path,
                    best_rebal_series_path, worst_rebal_series_path,
                    valid_prob=0.3,
                    batch_size=64, valid_batch_size=64,
                    window=250, eps=1e-6):
    """
        Get DataLoaders
    """
    features = np.load(features_path, allow_pickle=True)
    best_st_series = np.load(best_st_series_path, allow_pickle=True)
    worst_st_series = np.load(worst_st_series_path, allow_pickle=True)
    best_worst_pos_series =\
        np.load(best_worst_pos_series_path, allow_pickle=True)
    best_rebal_series =\
        np.load(best_rebal_series_path, allow_pickle=True)
    worst_rebal_series =\
        np.load(worst_rebal_series_path, allow_pickle=True)
    indices = best_worst_pos_series[:-1]

    valid_size = int(valid_prob * len(indices))
    valid_indices = np.random.choice(indices, valid_size, replace=False)

    train_indices = np.setdiff1d(indices, valid_indices)

    train_dataset =\
        InvestingStrategyGeneratorDataset(features, best_st_series,
                                        worst_st_series,
                                        best_worst_pos_series,
                                        best_rebal_series,
                                        worst_rebal_series,
                                        train_indices, window,
                                        eps)

    valid_dataset =\
        InvestingStrategyGeneratorDataset(features, best_st_series,
                                        worst_st_series,
                                        best_worst_pos_series,
                                        best_rebal_series,
                                        worst_rebal_series,
                                        valid_indices, window,
                                        eps)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=valid_batch_size, shuffle=True)

    return train_dataloader, valid_dataloader

def main():
    """
        main training method
    """
    args = _parser()

    options = dict(num_cpus=args.num_workers)
    ray.init(**options)

    config = GLEN_SCOTIA_TUNE_CONFIG

    config['epoch_size'] = args.epoch_size
    config['batch_size'] = args.batch_size
    config['valid_batch_size'] = args.valid_batch_size
    config['checkpoint_epoch'] = args.checkpoint_epoch
    config['num_samples'] = args.num_samples
    config['model_name'] = args.model_name
    config['device'] = args.device

    scheduler = ASHAScheduler(max_t=config['epoch_size'])

    stop = {"training_iteration": args.stop_training_iteration}

    if args.device == 'cpu':
        gpu_num = 0
    else:
        gpu_num = 1

    analysis = tune.run(
        train_glen_scotia,
        resources_per_trial={"cpu": args.num_workers,
                            "gpu": gpu_num},
        name=args.model_name,
        num_samples=args.num_samples,
        local_dir=config['checkpoint_dir'],
        stop=stop, metric="mean_loss", mode="min",
        scheduler=scheduler,
        config=config)

    best_checkpoint_dir = os.path.join(
        analysis.best_checkpoint, "checkpoint")

    model_state, optimizer_state = torch.load(best_checkpoint_dir)

    best_model_path = os.path.join(config['model_path'],
                                config['model_name']) + "/"

    if not os.path.exists(best_model_path):
        os.mkdir(best_model_path)

    torch.save(model_state,
            best_model_path + config['model_name'] + "_best.pt")
    torch.save(optimizer_state,
            best_model_path + config['model_name'] + "_best_optimizer.pt")

    print("Best hypermarameters found were: ", analysis.best_config)


def _parser():
    """
        argparse parser method

        Return:
            args
    """
    parser = argparse.ArgumentParser(
        description="argument parser for training Glen Scotia"
    )

    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Set number of workers for training")
    parser.add_argument(
        "--epoch_size", type=int, default=1000,
        help="epoch size")
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="batch size")
    parser.add_argument(
        "--valid_batch_size", type=int, default=32,
        help="valid batch size")
    parser.add_argument(
        "--checkpoint_epoch", type=int, default=3,
        help="checkpoint epoch")
    parser.add_argument(
        "--num_samples", type=int, default=5,
        help="num samples from ray tune")
    parser.add_argument(
        "--stop_training_iteration", type=int, default=100,
        help="stop training iteration for train")
    parser.add_argument(
        "--model_name", type=str, default="glen_scotia",
        help="model name")
    parser.add_argument(
        "--device", type=str, default='cuda',
        help="device")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()