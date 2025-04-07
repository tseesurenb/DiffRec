"""
Train a diffusion model for recommendation
"""

import argparse
from ast import parse
import os
import time
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import scipy.sparse as sp

import models.gaussian_diffusion as gd
from models.DNN import DNN
import evaluate_utils
from copy import deepcopy

import random
random_seed = 1
torch.manual_seed(random_seed) # cpu
torch.cuda.manual_seed(random_seed) # gpu
np.random.seed(random_seed) # numpy
random.seed(random_seed) # random and transforms
torch.backends.cudnn.deterministic=True # cudnn
def worker_init_fn(worker_id):
    np.random.seed(random_seed + worker_id)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

# Data loading and preprocessing functions
def load_interaction_data(file_path, skip_header=True):
    """
    Load interaction data from a text file with format: user_id item_id
    
    Args:
        file_path: path to the text file
        skip_header: whether to skip the first line (header)
        
    Returns:
        user_item_list: list of (user_id, item_id) pairs
        n_user: number of users
        n_item: number of items
    """
    user_item_list = []
    with open(file_path, 'r') as f:
        # Skip header if needed
        if skip_header:
            next(f)
            
        for line in f:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            parts = line.split()
            if len(parts) >= 2:  # Ensure we have at least user_id and item_id
                try:
                    user, item = int(parts[0]), int(parts[1])
                    user_item_list.append((user, item))
                except ValueError as e:
                    print(f"Skipping line '{line}': {e}")
    
    if not user_item_list:
        raise ValueError(f"No valid user-item interactions found in {file_path}")
        
    # Get dimensions
    n_user = max([x[0] for x in user_item_list]) + 1
    n_item = max([x[1] for x in user_item_list]) + 1
    
    return user_item_list, n_user, n_item

def create_sparse_matrix(user_item_list, n_user, n_item):
    """
    Create a sparse matrix from user-item interactions
    
    Args:
        user_item_list: list of (user_id, item_id) pairs
        n_user: number of users
        n_item: number of items
        
    Returns:
        sparse_matrix: scipy sparse matrix
    """
    sparse_matrix = sp.lil_matrix((n_user, n_item), dtype=np.float32)
    for user, item in user_item_list:
        sparse_matrix[user, item] = 1.0
    return sparse_matrix.tocsr()

def split_train_validation(train_user_item_list, validation_ratio=0.2, random_seed=42):
    """
    Split a user-item list into train and validation sets
    
    Args:
        train_user_item_list: list of (user_id, item_id) pairs
        validation_ratio: float, percentage of interactions to use for validation
        random_seed: int, random seed for reproducibility
        
    Returns:
        train_list, validation_list
    """
    np.random.seed(random_seed)
    
    # Create indices and shuffle
    indices = list(range(len(train_user_item_list)))
    np.random.shuffle(indices)
    
    # Split indices
    val_size = int(len(indices) * validation_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    # Create lists
    val_list = [train_user_item_list[i] for i in val_indices]
    train_list = [train_user_item_list[i] for i in train_indices]
    
    return train_list, val_list

class DataDiffusion(torch.utils.data.Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor
        
    def __getitem__(self, index):
        return self.data_tensor[index]
    
    def __len__(self):
        return self.data_tensor.shape[0]

# "yelp_clean_lr1e-05_wd0.0_bs400_dims[1000]_emb10_x0_steps5_scale0.01_min0.001_max0.01_sample0_reweight0_log.pth"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='yelp2018', help='choose the dataset')
parser.add_argument('--train_path', type=str, default='./data/yelp2018/train_coo.txt', help='path to training data file')
parser.add_argument('--test_path', type=str, default='./data/yelp2018/test_coo.txt', help='path to test data file')
parser.add_argument('--validation_ratio', type=float, default=0.2, help='ratio of training data to use as validation')
parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=400)
parser.add_argument('--epochs', type=int, default=1000, help='upper epoch limit')
parser.add_argument('--topN', type=str, default='[10, 20]')
parser.add_argument('--tst_w_val', action='store_true', help='test with validation')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--gpu', type=str, default='0', help='gpu card ID')
parser.add_argument('--save_path', type=str, default='./saved_models/', help='save model path')
parser.add_argument('--log_name', type=str, default='log', help='the log name')
parser.add_argument('--round', type=int, default=1, help='record the experiment')

# params for the model
parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
parser.add_argument('--dims', type=str, default='[2000]', help='the dims for the DNN')
parser.add_argument('--norm', type=bool, default=False, help='Normalize the input or not')
parser.add_argument('--emb_size', type=int, default=10, help='timestep embedding size')

# params for diffusion
parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
parser.add_argument('--steps', type=int, default=10, help='diffusion steps')
parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
parser.add_argument('--noise_scale', type=float, default=0.01, help='noise scale for noise generating')
parser.add_argument('--noise_min', type=float, default=0.001, help='noise lower bound for noise generating')
parser.add_argument('--noise_max', type=float, default=0.01, help='noise upper bound for noise generating')
parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
parser.add_argument('--sampling_steps', type=int, default=0, help='steps of the forward process during inference')
parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')

args = parser.parse_args()
print("args:", args)

#os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
#device = torch.device("cuda:0" if args.cuda else "cpu")

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

### DATA LOAD ###
print("Loading and processing data...")

# Load data from text files
train_user_item_list, n_user_train, n_item_train = load_interaction_data(args.train_path)
test_user_item_list, n_user_test, n_item_test = load_interaction_data(args.test_path)

# Get max dimensions
n_user = max(n_user_train, n_user_test)
n_item = max(n_item_train, n_item_test)

# Split training data into train and validation
train_list, valid_list = split_train_validation(
    train_user_item_list, args.validation_ratio, random_seed)

# Create sparse matrices
train_data = create_sparse_matrix(train_list, n_user, n_item)
valid_y_data = create_sparse_matrix(valid_list, n_user, n_item)
test_y_data = create_sparse_matrix(test_user_item_list, n_user, n_item)

# Create dataset and loader
train_dataset = DataDiffusion(torch.FloatTensor(train_data.toarray()))
train_loader = DataLoader(
    train_dataset, 
    batch_size=args.batch_size, 
    pin_memory=True,
    shuffle=True, 
    num_workers=0, 
    worker_init_fn=worker_init_fn
)
test_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

# Create TV dataset (train + validation) for testing with validation
if args.tst_w_val:
    tv_data = train_data + valid_y_data
    tv_dataset = DataDiffusion(torch.FloatTensor(tv_data.toarray()))
    test_twv_loader = DataLoader(tv_dataset, batch_size=args.batch_size, shuffle=False)
    mask_tv = tv_data
else:
    mask_tv = train_data + valid_y_data

print('Data ready.')
print(f'Number of users: {n_user}, Number of items: {n_item}')
print(f'Training interactions: {train_data.sum()}')
print(f'Validation interactions: {valid_y_data.sum()}')
print(f'Test interactions: {test_y_data.sum()}')

### Build Gaussian Diffusion ###
if args.mean_type == 'x0':
    mean_type = gd.ModelMeanType.START_X
elif args.mean_type == 'eps':
    mean_type = gd.ModelMeanType.EPSILON
else:
    raise ValueError("Unimplemented mean type %s" % args.mean_type)

diffusion = gd.GaussianDiffusion(mean_type, args.noise_schedule, \
        args.noise_scale, args.noise_min, args.noise_max, args.steps, device).to(device)

### Build MLP ###
out_dims = eval(args.dims) + [n_item]
in_dims = out_dims[::-1]
model = DNN(in_dims, out_dims, args.emb_size, time_type="cat", norm=args.norm).to(device)

optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
print("Models ready.")

param_num = 0
mlp_num = sum([param.nelement() for param in model.parameters()])
diff_num = sum([param.nelement() for param in diffusion.parameters()])  # 0
param_num = mlp_num + diff_num
print("Number of all parameters:", param_num)

def evaluate(data_loader, data_te, mask_his, topN):
    model.eval()
    e_idxlist = list(range(mask_his.shape[0]))
    e_N = mask_his.shape[0]

    predict_items = []
    target_items = []
    for i in range(e_N):
        target_items.append(data_te[i, :].nonzero()[1].tolist())
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            his_data = mask_his[e_idxlist[batch_idx*args.batch_size:batch_idx*args.batch_size+len(batch)]]
            batch = batch.to(device)
            prediction = diffusion.p_sample(model, batch, args.sampling_steps, args.sampling_noise)
            prediction[his_data.nonzero()] = -np.inf

            _, indices = torch.topk(prediction, topN[-1])
            indices = indices.cpu().numpy().tolist()
            predict_items.extend(indices)

    test_results = evaluate_utils.computeTopNAccuracy(target_items, predict_items, topN)

    return test_results

best_recall, best_epoch = -100, 0
best_test_result = None
print("Start training...")
for epoch in range(1, args.epochs + 1):
    if epoch - best_epoch >= 100:
        print('-'*18)
        print('Exiting from training early')
        break

    model.train()
    start_time = time.time()

    batch_count = 0
    total_loss = 0.0
    
    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)
        batch_count += 1
        optimizer.zero_grad()
        losses = diffusion.training_losses(model, batch, args.reweight)
        loss = losses["loss"].mean()
        total_loss += loss
        loss.backward()
        optimizer.step()
    
    if epoch % 5 == 0:
        valid_results = evaluate(test_loader, valid_y_data, train_data, eval(args.topN))
        if args.tst_w_val:
            test_results = evaluate(test_twv_loader, test_y_data, mask_tv, eval(args.topN))
        else:
            test_results = evaluate(test_loader, test_y_data, mask_tv, eval(args.topN))
        evaluate_utils.print_results(None, valid_results, test_results)

        if valid_results[1][1] > best_recall: # recall@20 as selection
            best_recall, best_epoch = valid_results[1][1], epoch
            best_results = valid_results
            best_test_results = test_results

            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            torch.save(model, '{}{}_lr{}_wd{}_bs{}_dims{}_emb{}_{}_steps{}_scale{}_min{}_max{}_sample{}_reweight{}_{}.pth' \
                .format(args.save_path, args.dataset, args.lr, args.weight_decay, args.batch_size, args.dims, args.emb_size, args.mean_type, \
                args.steps, args.noise_scale, args.noise_min, args.noise_max, args.sampling_steps, args.reweight, args.log_name))
    
    print("Running Epoch {:03d} ".format(epoch) + 'train loss {:.4f}'.format(total_loss) + " costs " + time.strftime(
                        "%H: %M: %S", time.gmtime(time.time()-start_time)))
    print('---'*18)

print('==='*18)
print("End. Best Epoch {:03d} ".format(best_epoch))
evaluate_utils.print_results(None, best_results, best_test_results)   
print("End time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))