"""
Train a diffusion model for recommendation with enhanced interaction features
"""

import argparse
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
from models.enhanced_dnn_model import EnhancedDNN
from enhanced_interaction_features import calculate_enhanced_interaction_features, EnhancedInteractionDataset, calculate_enhanced_interaction_features_batched, calculate_enhanced_interaction_features_batched_topk
import evaluate_utils
import data_utils
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

# ml-1m_clean_lr0.001_wd0.0_bs400_dims[200,600]_emb10_x0_steps40_scale0.005_min0.005_max0.01_sample0_reweight1_log.pth
# "amazon-book_clean_lr5e-05_wd0.0_bs400_dims[1000]_emb10_x0_steps5_scale0.0001_min0.0005_max0.005_sample0_reweight0_log.pth"

# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', type=str, default='amazon-book_clean', help='choose the dataset')
# parser.add_argument('--data_path', type=str, default='./data/', help='load data path')
# parser.add_argument('--lr', type=float, default=0.00005, help='learning rate')
# parser.add_argument('--weight_decay', type=float, default=0.0)
# parser.add_argument('--batch_size', type=int, default=400)
# parser.add_argument('--epochs', type=int, default=1000, help='upper epoch limit')
# parser.add_argument('--topN', type=str, default='[10, 20]')
# parser.add_argument('--tst_w_val', action='store_true', help='test with validation')
# parser.add_argument('--cuda', action='store_true', help='use CUDA')
# parser.add_argument('--gpu', type=str, default='0', help='gpu card ID')
# parser.add_argument('--save_path', type=str, default='./saved_models/', help='save model path')
# parser.add_argument('--log_name', type=str, default='log', help='the log name')
# parser.add_argument('--round', type=int, default=1, help='record the experiment')

# # params for the model
# parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
# parser.add_argument('--dims', type=str, default='[1000]', help='the dims for the DNN')
# parser.add_argument('--norm', type=bool, default=False, help='Normalize the input or not')
# parser.add_argument('--emb_size', type=int, default=10, help='timestep embedding size')

# # params for diffusion
# parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
# parser.add_argument('--steps', type=int, default=5, help='diffusion steps')
# parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
# parser.add_argument('--noise_scale', type=float, default=0.0001, help='noise scale for noise generating')
# parser.add_argument('--noise_min', type=float, default=0.0005, help='noise lower bound for noise generating')
# parser.add_argument('--noise_max', type=float, default=0.005, help='noise upper bound for noise generating')
# parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
# parser.add_argument('--sampling_steps', type=int, default=0, help='steps of the forward process during inference')
# parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')

# "yelp_clean_lr1e-05_wd0.0_bs400_dims[1000]_emb10_x0_steps5_scale0.01_min0.001_max0.01_sample0_reweight0_log.pth"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='yelp_clean', help='choose the dataset')
parser.add_argument('--data_path', type=str, default='./data/', help='load data path')
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
parser.add_argument('--dims', type=str, default='[1000]', help='the dims for the DNN')
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


# ml-1m_clean_lr0.001_wd0.0_bs400_dims[200,600]_emb10_x0_steps40_scale0.005_min0.005_max0.01_sample0_reweight1_log.pth

# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', type=str, default='ml-1m_clean', help='choose the dataset')
# parser.add_argument('--data_path', type=str, default='./data/', help='load data path')
# parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
# parser.add_argument('--weight_decay', type=float, default=0.0)
# parser.add_argument('--batch_size', type=int, default=400)
# parser.add_argument('--epochs', type=int, default=1000, help='upper epoch limit')
# parser.add_argument('--topN', type=str, default='[10, 20]')
# parser.add_argument('--tst_w_val', action='store_true', help='test with validation')
# parser.add_argument('--cuda', action='store_true', help='use CUDA')
# parser.add_argument('--gpu', type=str, default='0', help='gpu card ID')
# parser.add_argument('--save_path', type=str, default='./saved_models/', help='save model path')
# parser.add_argument('--log_name', type=str, default='log', help='the log name')
# parser.add_argument('--round', type=int, default=1, help='record the experiment')

# # params for the model
# parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
# parser.add_argument('--dims', type=str, default='[200, 600]', help='the dims for the DNN')
# parser.add_argument('--norm', type=bool, default=False, help='Normalize the input or not')
# parser.add_argument('--emb_size', type=int, default=10, help='timestep embedding size')

# # params for diffusion
# parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
# parser.add_argument('--steps', type=int, default=40, help='diffusion steps')
# parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
# parser.add_argument('--noise_scale', type=float, default=0.005, help='noise scale for noise generating')
# parser.add_argument('--noise_min', type=float, default=0.005, help='noise lower bound for noise generating')
# parser.add_argument('--noise_max', type=float, default=0.01, help='noise upper bound for noise generating')
# parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
# parser.add_argument('--sampling_steps', type=int, default=0, help='steps of the forward process during inference')
# parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')


args = parser.parse_args()
print("args:", args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda:0" if args.cuda else "cpu")

print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

### DATA LOAD ###
print("Loading data...")
train_path = args.data_path + args.dataset + '/train_list.npy'
valid_path = args.data_path + args.dataset + '/valid_list.npy'
test_path = args.data_path + args.dataset + '/test_list.npy'

# Load interaction data
train_data, valid_y_data, test_y_data, n_user, n_item = data_utils.data_load(train_path, valid_path, test_path)

print(f"Data loaded: {n_user} users, {n_item} items")
print(f"Train interactions: {train_data.sum()}")
print(f"Valid interactions: {valid_y_data.sum()}")
print(f"Test interactions: {test_y_data.sum()}")

# Calculate enhanced interaction features for training data
#item_counts_matrix, co_counts_matrix = calculate_enhanced_interaction_features(train_data)
item_counts_matrix, co_counts_matrix = calculate_enhanced_interaction_features_batched(train_data)

item_counts_matrix, co_counts_matrix = calculate_enhanced_interaction_features_batched_topk(train_data, batch_size=5000, k=100)

print("Converting to tensors...")
# Convert to tensor format for the dataset
train_tensor = torch.FloatTensor(train_data.toarray())
item_counts_tensor = torch.FloatTensor(item_counts_matrix)
co_counts_tensor = torch.FloatTensor(co_counts_matrix.toarray())

# Create dataset and dataloader with enhanced features
train_dataset = EnhancedInteractionDataset(train_tensor, item_counts_tensor, co_counts_tensor)
train_loader = DataLoader(
    train_dataset, 
    batch_size=args.batch_size, 
    pin_memory=True, 
    shuffle=True, 
    num_workers=0, 
    worker_init_fn=worker_init_fn
)

# Create test loader
test_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

# For testing with validation if enabled
if args.tst_w_val:
    # Combine train and validation data
    tv_data = train_data + valid_y_data
    
    # Calculate enhanced features for combined data
    tv_item_counts, tv_co_counts = calculate_enhanced_interaction_features(tv_data)
    
    # Convert to tensors
    tv_tensor = torch.FloatTensor(tv_data.toarray())
    tv_item_counts_tensor = torch.FloatTensor(tv_item_counts)
    tv_co_counts_tensor = torch.FloatTensor(tv_co_counts.toarray())
    
    # Create dataset and loader
    tv_dataset = EnhancedInteractionDataset(tv_tensor, tv_item_counts_tensor, tv_co_counts_tensor)
    test_twv_loader = DataLoader(tv_dataset, batch_size=args.batch_size, shuffle=False)

mask_tv = train_data + valid_y_data

print('Data preparation complete.')

### Build Gaussian Diffusion ###
if args.mean_type == 'x0':
    mean_type = gd.ModelMeanType.START_X
elif args.mean_type == 'eps':
    mean_type = gd.ModelMeanType.EPSILON
else:
    raise ValueError(f"Unimplemented mean type {args.mean_type}")

diffusion = gd.GaussianDiffusion(mean_type, args.noise_schedule, 
                               args.noise_scale, args.noise_min, 
                               args.noise_max, args.steps, device).to(device)

### Build Enhanced MLP ###
out_dims = eval(args.dims) + [n_item]
in_dims = out_dims[::-1]
model = EnhancedDNN(in_dims, out_dims, args.emb_size, 
                   time_type="cat", norm=args.norm).to(device)

optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
print("Models ready.")

param_num = sum([param.nelement() for param in model.parameters()])
print(f"Number of parameters: {param_num}")

# Modified version of p_sample for the enhanced model
def p_sample_with_enhanced_features(model, diffusion, x_start, item_counts, co_counts, steps, sampling_noise=False):
    """Custom p_sample function that includes enhanced interaction features"""
    assert steps <= diffusion.steps, "Too many steps in inference."
    
    if steps == 0:
        x_t = x_start
    else:
        t = torch.tensor([steps - 1] * x_start.shape[0]).to(x_start.device)
        x_t = diffusion.q_sample(x_start, t)

    indices = list(range(diffusion.steps))[::-1]

    for i in indices:
        t = torch.tensor([i] * x_t.shape[0]).to(x_start.device)
        
        # Calculate model output with enhanced features
        model_output = model(x_t, t, item_counts, co_counts)
        
        # If using the standard diffusion approach (not skipping noise)
        if diffusion.noise_scale != 0.:
            # Get mean and variance for the step
            out = {}
            if diffusion.mean_type == gd.ModelMeanType.START_X:
                pred_xstart = model_output
            elif diffusion.mean_type == gd.ModelMeanType.EPSILON:
                pred_xstart = diffusion._predict_xstart_from_eps(x_t, t, eps=model_output)
            
            # Get posterior mean and variance
            posterior_mean, posterior_variance, posterior_log_variance = diffusion.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x_t, t=t
            )
            
            out["mean"] = posterior_mean
            out["log_variance"] = posterior_log_variance
            
            if sampling_noise:
                noise = torch.randn_like(x_t)
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
                )  # no noise when t == 0
                x_t = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
            else:
                x_t = out["mean"]
        else:
            # For the case where we skip noise (special case)
            x_t = model_output
            
    return x_t

def evaluate(data_loader, data_te, mask_his, topN):
    """Evaluate the model performance"""
    model.eval()
    e_idxlist = list(range(mask_his.shape[0]))
    e_N = mask_his.shape[0]

    predict_items = []
    target_items = []
    for i in range(e_N):
        target_items.append(data_te[i, :].nonzero()[1].tolist())
    
    with torch.no_grad():
        for batch_idx, (batch, item_counts, co_counts) in enumerate(data_loader):
            his_data = mask_his[e_idxlist[batch_idx*args.batch_size:batch_idx*args.batch_size+len(batch)]]
            batch = batch.to(device)
            item_counts = item_counts.to(device)
            co_counts = co_counts.to(device)
            
            # Use the modified p_sample function with enhanced features
            prediction = p_sample_with_enhanced_features(
                model, diffusion, batch, item_counts, co_counts, 
                args.sampling_steps, args.sampling_noise
            )
            
            # Mask out already interacted items
            prediction[his_data.nonzero()] = -np.inf

            # Get top-K items
            _, indices = torch.topk(prediction, topN[-1])
            indices = indices.cpu().numpy().tolist()
            predict_items.extend(indices)

    # Calculate metrics
    test_results = evaluate_utils.computeTopNAccuracy(target_items, predict_items, topN)
    return test_results

# Custom training losses function for the diffusion model with enhanced features
def custom_training_losses(model, diffusion, x_start, item_counts, co_counts, reweight=False):
    """Custom training_losses function that includes enhanced interaction features"""
    batch_size, device = x_start.size(0), x_start.device
    
    # Sample timesteps (same as in the original diffusion model)
    ts, pt = diffusion.sample_timesteps(batch_size, device, 'importance')
    
    # Sample noise
    noise = torch.randn_like(x_start)
    
    # Get noisy sample
    if diffusion.noise_scale != 0.:
        x_t = diffusion.q_sample(x_start, ts, noise)
    else:
        x_t = x_start

    # Get model prediction with enhanced features
    model_output = model(x_t, ts, item_counts, co_counts)
    
    # Determine target based on mean type
    target = {
        gd.ModelMeanType.START_X: x_start,
        gd.ModelMeanType.EPSILON: noise,
    }[diffusion.mean_type]

    assert model_output.shape == target.shape == x_start.shape

    # Calculate MSE loss
    mse = torch.mean((target - model_output) ** 2, dim=-1)

    # Apply reweighting if enabled
    if reweight:
        if diffusion.mean_type == gd.ModelMeanType.START_X:
            weight = diffusion.SNR(ts - 1) - diffusion.SNR(ts)
            weight = torch.where((ts == 0), torch.ones_like(weight), weight)
            loss = mse
        elif diffusion.mean_type == gd.ModelMeanType.EPSILON:
            weight = (1 - diffusion.alphas_cumprod[ts]) / ((1-diffusion.alphas_cumprod_prev[ts])**2 * (1-diffusion.betas[ts]))
            weight = torch.where((ts == 0), torch.ones_like(weight), weight)
            likelihood = torch.mean((x_start - diffusion._predict_xstart_from_eps(x_t, ts, model_output))**2, dim=-1) / 2.0
            loss = torch.where((ts == 0), likelihood, mse)
    else:
        weight = torch.ones_like(mse)
        loss = mse

    # Prepare the losses dictionary
    terms = {}
    terms["loss"] = weight * loss
    
    # Update Lt_history for importance sampling (same as original)
    for t, loss_val in zip(ts, terms["loss"]):
        if diffusion.Lt_count[t] == diffusion.history_num_per_term:
            Lt_history_old = diffusion.Lt_history.clone()
            diffusion.Lt_history[t, :-1] = Lt_history_old[t, 1:]
            diffusion.Lt_history[t, -1] = loss_val.detach()
        else:
            try:
                diffusion.Lt_history[t, diffusion.Lt_count[t]] = loss_val.detach()
                diffusion.Lt_count[t] += 1
            except Exception as e:
                print(f"Error updating Lt_history: {e}")
                print(f"t: {t}, Lt_count[t]: {diffusion.Lt_count[t]}, loss_val: {loss_val}")
                raise ValueError

    # Apply importance sampling correction
    terms["loss"] /= pt
    return terms

# Training loop
best_recall, best_epoch = -100, 0
best_test_result = None
print("Start training...")
for epoch in range(1, args.epochs + 1):
    if epoch - best_epoch >= 20:
        print('-'*18)
        print('Exiting from training early')
        break

    model.train()
    start_time = time.time()

    batch_count = 0
    total_loss = 0.0
    
    for batch_idx, (batch, item_counts, co_counts) in enumerate(train_loader):
        batch = batch.to(device)
        item_counts = item_counts.to(device)
        co_counts = co_counts.to(device)
        
        batch_count += 1
        optimizer.zero_grad()
        
        # Use custom training loss function with enhanced features
        losses = custom_training_losses(model, diffusion, batch, item_counts, co_counts, args.reweight)
        loss = losses["loss"].mean()
        
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    
    if epoch % 5 == 0:
        print(f"Evaluating at epoch {epoch}...")
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
            
            # Save model
            model_path = f'{args.save_path}{args.dataset}_enhanced_lr{args.lr}_wd{args.weight_decay}_bs{args.batch_size}_' + \
                         f'dims{args.dims}_emb{args.emb_size}_{args.mean_type}_steps{args.steps}_' + \
                         f'scale{args.noise_scale}_min{args.noise_min}_max{args.noise_max}_' + \
                         f'sample{args.sampling_steps}_reweight{args.reweight}_{args.log_name}.pth'
            
            torch.save(model, model_path)
            print(f"Saved best model to {model_path}")
    
    print(f"Epoch {epoch:03d} " + f'train loss {avg_loss:.4f}' + " costs " + time.strftime(
                        "%H: %M: %S", time.gmtime(time.time()-start_time)))
    print('---'*18)

print('==='*18)
print(f"End. Best Epoch {best_epoch:03d}")
evaluate_utils.print_results(None, best_results, best_test_results)   
print("End time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))