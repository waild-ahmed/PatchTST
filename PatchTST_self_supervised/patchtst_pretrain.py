import numpy as np
import pandas as pd
import os
import torch
from torch import nn
import glob

from src.models.patchTST import PatchTST
from src.learner import Learner, transfer_weights
from src.callback.tracking import *
from src.callback.patch_mask import *
from src.callback.transforms import *
from src.metrics import *
from src.basics import set_device
from datautils import *


import argparse

parser = argparse.ArgumentParser()
# Dataset and dataloader
parser.add_argument('--dataset_dir', type=str, default='datasets', help='directory containing CSV files')
parser.add_argument('--context_points', type=int, default=512, help='sequence length')
parser.add_argument('--target_points', type=int, default=96, help='forecast horizon')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for DataLoader')
parser.add_argument('--scaler', type=str, default='standard', help='scale the input data')
parser.add_argument('--features', type=str, default='M', help='for multivariate model or univariate model')
# Patch
parser.add_argument('--patch_len', type=int, default=12, help='patch length')
parser.add_argument('--stride', type=int, default=12, help='stride between patch')
# RevIN
parser.add_argument('--revin', type=int, default=1, help='reversible instance normalization')
# Model args
parser.add_argument('--n_layers', type=int, default=3, help='number of Transformer layers')
parser.add_argument('--n_heads', type=int, default=16, help='number of Transformer heads')
parser.add_argument('--d_model', type=int, default=128, help='Transformer d_model')
parser.add_argument('--d_ff', type=int, default=512, help='Tranformer MLP dimension')
parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
parser.add_argument('--head_dropout', type=float, default=0.2, help='head dropout')
# Pretrain mask
parser.add_argument('--mask_ratio', type=float, default=0.4, help='masking ratio for the input')
# Optimization args
parser.add_argument('--n_epochs_pretrain', type=int, default=10, help='number of pre-training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
# model id to keep track of the number of models saved
parser.add_argument('--pretrained_model_id', type=int, default=1, help='id of the saved pretrained model')
parser.add_argument('--model_type', type=str, default='based_model', help='for multivariate model or univariate model')


args = parser.parse_args()
print('args:', args)
args.save_pretrained_model = 'patchtst_pretrained_cw'+str(args.context_points)+'_patch'+str(args.patch_len) + '_stride'+str(args.stride) + '_epochs-pretrain' + str(args.n_epochs_pretrain) + '_mask' + str(args.mask_ratio)  + '_model' + str(args.pretrained_model_id)
args.save_path = 'saved_models/multi_stock/masked_patchtst/' + args.model_type + '/'
if not os.path.exists(args.save_path): os.makedirs(args.save_path)


# get available GPU device
set_device()


class MultiStockDataset(Dataset):
    _cached_data = None  # Class variable to store cached data
    
    def __init__(self, root_path, split='train', context_points=512, target_points=96, features='M'):
        self.root_path = root_path
        self.context_points = context_points
        self.target_points = target_points
        self.features = features
        self.split = split
        
        if MultiStockDataset._cached_data is None:
            print("\nLoading and processing data for the first time...")
            self._load_and_process_data()
        else:
            print("\nUsing cached data...")
            self.data = MultiStockDataset._cached_data
            
        # Split data into train/val/test
        n = len(self.data)
        if split == 'train':
            self.data = self.data[:int(0.7 * n)]
        elif split == 'val':
            self.data = self.data[int(0.7 * n):int(0.9 * n)]
        else:  # test
            self.data = self.data[int(0.9 * n):]
        
        print(f"{split} set size: {len(self.data)}")
            
    def _load_and_process_data(self):
        # Load all CSV files from the directory
        self.csv_files = glob.glob(os.path.join(self.root_path, "*_daily.csv"))
        if not self.csv_files:
            raise ValueError(f"No CSV files found in {self.root_path}")
        print(f"Found {len(self.csv_files)} CSV files")
        
        self.data = []
        self.scalers = {}
        
        for file in self.csv_files:
            try:
                print(f"\nProcessing {os.path.basename(file)}")
                df = pd.read_csv(file)
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                print(f"Numeric columns: {list(numeric_cols)}")
                
                if len(numeric_cols) == 0:
                    print(f"Skipping {file} - no numeric columns found")
                    continue
                
                # Scale the data
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(df[numeric_cols])
                self.scalers[file] = scaler
                
                # Convert to sliding windows
                n_windows = len(scaled_data) - self.context_points - self.target_points + 1
                if n_windows <= 0:
                    print(f"Skipping {file} - insufficient data points")
                    continue
                    
                print(f"Creating {n_windows} windows from {len(scaled_data)} data points")
                for i in range(0, n_windows):
                    window = scaled_data[i:i + self.context_points + self.target_points]
                    self.data.append(window)
                
                print(f"Successfully processed {os.path.basename(file)}")
                
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                continue
        
        self.data = np.array(self.data)
        print(f"\nTotal windows created: {len(self.data)}")
        print(f"Data shape: {self.data.shape}")
        
        # Cache the data
        MultiStockDataset._cached_data = self.data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        window = self.data[idx]
        x = window[:self.context_points]
        y = window[self.context_points:]
        return torch.FloatTensor(x), torch.FloatTensor(y)


def get_dls(args):
    """Get train and validation dataloaders"""
    # Create DataLoaders with the correct initialization parameters
    dls = DataLoaders(
        datasetCls=MultiStockDataset,
        dataset_kwargs={
            'root_path': args.dataset_dir,
            'context_points': args.context_points,
            'target_points': args.target_points,
            'features': args.features
        },
        batch_size=args.batch_size,
        workers=args.num_workers,
        shuffle_train=True,
        shuffle_val=False
    )
    
    return dls


def get_model(c_in, args):
    """
    c_in: number of variables
    """
    # get number of patches
    num_patch = (max(args.context_points, args.patch_len)-args.patch_len) // args.stride + 1    
    print('number of patches:', num_patch)
    
    # get model
    model = PatchTST(c_in=c_in,
                target_dim=args.target_points,
                patch_len=args.patch_len,
                stride=args.stride,
                num_patch=num_patch,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                d_model=args.d_model,
                shared_embedding=True,
                d_ff=args.d_ff,                        
                dropout=args.dropout,
                head_dropout=args.head_dropout,
                act='relu',
                head_type='pretrain',
                res_attention=False
                )        
    # print out the model size
    print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model


def find_lr():
    # get dataloader
    dls = get_dls(args)    
    model = get_model(dls.train.dataset.data.shape[2], args)  # Use the number of features from the dataset
    # get loss
    loss_func = torch.nn.MSELoss(reduction='mean')
    # get callbacks
    cbs = [RevInCB(dls.train.dataset.data.shape[2], denorm=False)] if args.revin else []
    cbs += [PatchMaskCB(patch_len=args.patch_len, stride=args.stride, mask_ratio=args.mask_ratio)]
        
    # define learner
    learn = Learner(dls, model, 
                    loss_func, 
                    lr=args.lr, 
                    cbs=cbs,
                    )                        
    
    # Instead of using lr_finder, we'll use a fixed learning rate
    print(f"Using fixed learning rate: {args.lr}")
    return args.lr


def pretrain_func(lr=args.lr):
    print("\nStarting pretraining...")
    # get dataloader
    dls = get_dls(args)
    print(f"Train set size: {len(dls.train.dataset)}, Val set size: {len(dls.valid.dataset)}")
    
    # get model     
    model = get_model(dls.train.dataset.data.shape[2], args)  # Use the number of features from the dataset
    print("Model created successfully")
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")
    
    # get loss
    loss_func = torch.nn.MSELoss(reduction='mean')
    
    # get callbacks with progress tracking
    cbs = [RevInCB(dls.train.dataset.data.shape[2], denorm=False)] if args.revin else []
    cbs += [
         PatchMaskCB(patch_len=args.patch_len, stride=args.stride, mask_ratio=args.mask_ratio),
         SaveModelCB(monitor='valid_loss', fname=args.save_pretrained_model,                       
                        path=args.save_path)
        ]
    
    # define learner
    learn = Learner(dls, model, 
                    loss_func, 
                    lr=lr, 
                    cbs=cbs,
                    )
    
    print("\nStarting training loop...")
    print(f"Training for {args.n_epochs_pretrain} epochs")
    
    try:
        # fit the data to the model with progress tracking
        learn.fit_one_cycle(n_epochs=args.n_epochs_pretrain, lr_max=lr)
        print("Training completed successfully")

        # Save training metrics
        train_loss = learn.recorder['train_loss']
        valid_loss = learn.recorder['valid_loss']
        df = pd.DataFrame(data={'train_loss': train_loss, 'valid_loss': valid_loss})
        save_path = args.save_path + args.save_pretrained_model + '_losses.csv'
        df.to_csv(save_path, float_format='%.6f', index=False)
        print(f"Training losses saved to {save_path}")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e


if __name__ == '__main__':
    print("Starting script...")
    print(f"Arguments: {args}")
    
    # Create save directory if it doesn't exist
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print(f"Created save directory: {args.save_path}")
    
    # Skip learning rate finding and use fixed learning rate
    lr = args.lr
    print(f"Using learning rate: {lr}")
    
    try:
        # Pretrain
        pretrain_func(lr)
        print('Pretraining completed')
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during script execution: {str(e)}")
        import traceback
        traceback.print_exc()