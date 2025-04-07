"""
Train a diffusion model for recommendation with contrastive learning and hyperparameter search
"""

import argparse
import os
import time
import numpy as np
import copy
import itertools
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import models.gaussian_diffusion as gd
from models.DNN import DNN
from models.DiffRecContrastive import DiffRecContrastive
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

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ml-1m_clean', help='choose the dataset')
parser.add_argument('--data_path', type=str, default='./datasets/', help='load data path')
parser.add_argument('--lr', type=str, default='[0.001]', help='learning rates to search')
parser.add_argument('--weight_decay', type=str, default='[0.0]', help='weight decay values to search')
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
parser.add_argument('--dims', type=str, default='[[200,600]]', help='dimensions for the DNN to search')
parser.add_argument('--norm', type=bool, default=False, help='Normalize the input or not')
parser.add_argument('--emb_size', type=str, default='[10]', help='timestep embedding sizes to search')

# params for diffusion
parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
parser.add_argument('--steps', type=str, default='[40]', help='diffusion steps to search')
parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
parser.add_argument('--noise_scale', type=str, default='[0.005]', help='noise scales to search')
parser.add_argument('--noise_min', type=str, default='[0.005]', help='noise lower bounds to search')
parser.add_argument('--noise_max', type=str, default='[0.01]', help='noise upper bounds to search')
parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
parser.add_argument('--sampling_steps', type=int, default=0, help='steps of the forward process during inference')
parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')

# params for contrastive learning
parser.add_argument('--contrastive', action='store_true', help='use contrastive learning')
parser.add_argument('--temperature', type=str, default='[0.1]', help='temperatures for contrastive loss to search')
parser.add_argument('--contrastive_weight', type=str, default='[0.2]', help='weights for contrastive loss to search')
parser.add_argument('--neg_samples', type=str, default='[1]', help='numbers of negative samples to search')

# params for hyperparameter search
parser.add_argument('--search', action='store_true', help='perform hyperparameter search')
parser.add_argument('--max_trials', type=int, default=10, help='maximum number of hyperparameter combinations to try')
parser.add_argument('--early_stop', type=int, default=20, help='early stopping patience for each trial')
parser.add_argument('--eval_interval', type=int, default=5, help='epochs between evaluations')
parser.add_argument('--results_file', type=str, default='hparam_results.json', help='file to save hyperparameter search results')

args = parser.parse_args()
print("args:", args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda:0" if args.cuda else "cpu")

print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

### DATA LOAD ###
train_path = args.data_path + 'train_list.npy'
valid_path = args.data_path + 'valid_list.npy'
test_path = args.data_path + 'test_list.npy'

train_data, valid_y_data, test_y_data, n_user, n_item = data_utils.data_load(train_path, valid_path, test_path)
train_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.A))
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=0, worker_init_fn=worker_init_fn)
test_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

if args.tst_w_val:
    tv_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.A) + torch.FloatTensor(valid_y_data.A))
    test_twv_loader = DataLoader(tv_dataset, batch_size=args.batch_size, shuffle=False)
mask_tv = train_data + valid_y_data

print('data ready.')

def sample_negative_items(batch, n_item, num_samples=1):
    """Sample negative items for users in the batch"""
    users = []
    pos_items = []
    neg_items = []
    
    for i in range(batch.shape[0]):
        user_idx = i  # User index within the batch
        
        # Get positive items for this user
        user_pos_items = batch[i].nonzero().squeeze(1).tolist()
        if not user_pos_items:  # Skip users with no positive items
            continue
        
        # If there's only one dimension in the result from nonzero(), convert to list
        if not isinstance(user_pos_items, list):
            user_pos_items = [user_pos_items]
            
        # Choose a random positive item
        pos_idx = random.choice(user_pos_items)
        
        # Sample negative items (items the user hasn't interacted with)
        all_items = set(range(n_item))
        negative_items = list(all_items - set(user_pos_items))
        
        # If no negatives available, skip this user
        if not negative_items:
            continue
            
        # Sample negative items
        sampled_neg = random.sample(negative_items, min(num_samples, len(negative_items)))
        
        for neg in sampled_neg:
            users.append(user_idx)
            pos_items.append(pos_idx)
            neg_items.append(neg)
    
    return torch.tensor(users, device=batch.device), torch.tensor(pos_items, device=batch.device), torch.tensor(neg_items, device=batch.device)

def evaluate(model, diffusion, data_loader, data_te, mask_his, topN, args):
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

def train_model(hparams, early_stop=20, eval_interval=5, max_epochs=1000):
    """Train a model with the given hyperparameters and return validation metrics"""
    # Build Gaussian Diffusion
    if args.mean_type == 'x0':
        mean_type = gd.ModelMeanType.START_X
    elif args.mean_type == 'eps':
        mean_type = gd.ModelMeanType.EPSILON
    else:
        raise ValueError(f"Unimplemented mean type {args.mean_type}")

    diffusion = gd.GaussianDiffusion(mean_type, args.noise_schedule, 
                                    hparams['noise_scale'], hparams['noise_min'], 
                                    hparams['noise_max'], hparams['steps'], device).to(device)

    # Build MLP
    out_dims = hparams['dims'] + [n_item]
    in_dims = out_dims[::-1]

    if args.contrastive:
        # Use contrastive-enabled model
        model = DiffRecContrastive(in_dims, out_dims, hparams['emb_size'], n_user, n_item, 
                                 hparams['temperature'], hparams['contrastive_weight'], 
                                 time_type="cat", norm=args.norm).to(device)
        print(f"Training with contrastive learning: temp={hparams['temperature']}, weight={hparams['contrastive_weight']}")
    else:
        # Use standard model
        model = DNN(in_dims, out_dims, hparams['emb_size'], time_type="cat", norm=args.norm).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=hparams['lr'], weight_decay=hparams['weight_decay'])
    
    param_num = sum([param.nelement() for param in model.parameters()])
    print(f"Number of parameters: {param_num}")
    
    # Training loop
    best_recall, best_epoch = -100, 0
    best_valid_result = None
    best_test_result = None
    
    print("Start training...")
    for epoch in range(1, max_epochs + 1):
        if epoch - best_epoch >= early_stop:
            print('-'*18)
            print('Exiting from training early')
            break

        model.train()
        start_time = time.time()

        batch_count = 0
        total_loss = 0.0
        total_diff_loss = 0.0
        total_cont_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            batch_count += 1
            optimizer.zero_grad()
            
            # Standard diffusion loss
            losses = diffusion.training_losses(model, batch, args.reweight)
            loss = losses["loss"].mean()
            diff_loss = loss.item()
            
            # Add contrastive loss if enabled
            if args.contrastive:
                # Sample negative items
                users, pos_items, neg_items = sample_negative_items(batch, n_item, hparams['neg_samples'])
                
                if len(users) > 0:  # Only compute contrastive loss if we have valid samples
                    cont_loss = model.contrastive_loss(users, pos_items, neg_items)
                    loss = loss + hparams['contrastive_weight'] * cont_loss
                    total_cont_loss += cont_loss.item()
            
            total_loss += loss.item()
            total_diff_loss += diff_loss
            
            loss.backward()
            optimizer.step()
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        
        if epoch % eval_interval == 0:
            valid_results = evaluate(model, diffusion, test_loader, valid_y_data, train_data, eval(args.topN), args)
            if args.tst_w_val:
                test_results = evaluate(model, diffusion, test_twv_loader, test_y_data, mask_tv, eval(args.topN), args)
            else:
                test_results = evaluate(model, diffusion, test_loader, test_y_data, mask_tv, eval(args.topN), args)
            
            # Print current results
            print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")
            evaluate_utils.print_results(None, valid_results, test_results)

            if valid_results[1][1] > best_recall: # recall@20 as selection criterion
                best_recall, best_epoch = valid_results[1][1], epoch
                best_valid_result = valid_results
                best_test_result = test_results
                
                # Save best model for this trial
                if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)
                    
                # Construct model name with all relevant hyperparameters
                model_name = f"{args.dataset}_lr{hparams['lr']}_wd{hparams['weight_decay']}_bs{args.batch_size}_"
                model_name += f"dims{hparams['dims']}_emb{hparams['emb_size']}_{args.mean_type}_steps{hparams['steps']}_"
                model_name += f"scale{hparams['noise_scale']}_min{hparams['noise_min']}_max{hparams['noise_max']}_"
                model_name += f"sample{args.sampling_steps}_reweight{args.reweight}"
                
                if args.contrastive:
                    model_name += f"_cont{hparams['contrastive_weight']}_temp{hparams['temperature']}_neg{hparams['neg_samples']}"
                    
                model_name += f"_{args.log_name}.pth"
                save_path = os.path.join(args.save_path, model_name)
                torch.save(model, save_path)
                print(f"Model saved to {save_path}")
        
        print(f"Epoch {epoch} completed in {time.strftime('%H:%M:%S', time.gmtime(time.time()-start_time))}")
        print('---'*18)

    print("Training completed.")
    print(f"Best epoch: {best_epoch}")
    evaluate_utils.print_results(None, best_valid_result, best_test_result)
    
    # Return metrics for hyperparameter optimization
    return {
        'best_epoch': best_epoch,
        'best_recall': best_recall,
        'valid_precision': best_valid_result[0],
        'valid_recall': best_valid_result[1],
        'valid_ndcg': best_valid_result[2],
        'valid_mrr': best_valid_result[3],
        'test_precision': best_test_result[0],
        'test_recall': best_test_result[1],
        'test_ndcg': best_test_result[2],
        'test_mrr': best_test_result[3],
    }

def hyperparameter_search():
    """Perform grid search over hyperparameters"""
    # Parse hyperparameter ranges from arguments
    learning_rates = eval(args.lr)
    weight_decays = eval(args.weight_decay)
    emb_sizes = eval(args.emb_size)
    dims_list = eval(args.dims)
    steps_list = eval(args.steps)
    noise_scales = eval(args.noise_scale)
    noise_mins = eval(args.noise_min)
    noise_maxs = eval(args.noise_max)
    
    # Contrastive learning parameters
    temperatures = eval(args.temperature) if args.contrastive else [0.1]
    contrastive_weights = eval(args.contrastive_weight) if args.contrastive else [0.2]
    neg_samples_list = eval(args.neg_samples) if args.contrastive else [1]
    
    # Build hyperparameter grid
    param_grid = []
    
    if args.contrastive:
        # With contrastive learning
        for lr, wd, emb, dims, steps, ns, nm, nx, temp, cw, neg in itertools.product(
            learning_rates, weight_decays, emb_sizes, dims_list, steps_list, 
            noise_scales, noise_mins, noise_maxs, temperatures, contrastive_weights, neg_samples_list
        ):
            param_grid.append({
                'lr': lr,
                'weight_decay': wd,
                'emb_size': emb,
                'dims': dims,
                'steps': steps,
                'noise_scale': ns,
                'noise_min': nm,
                'noise_max': nx,
                'temperature': temp,
                'contrastive_weight': cw,
                'neg_samples': neg
            })
    else:
        # Without contrastive learning
        for lr, wd, emb, dims, steps, ns, nm, nx in itertools.product(
            learning_rates, weight_decays, emb_sizes, dims_list, steps_list, 
            noise_scales, noise_mins, noise_maxs
        ):
            param_grid.append({
                'lr': lr,
                'weight_decay': wd,
                'emb_size': emb,
                'dims': dims,
                'steps': steps,
                'noise_scale': ns,
                'noise_min': nm,
                'noise_max': nx,
                'temperature': 0.1,  # Default values for non-contrastive
                'contrastive_weight': 0.0,
                'neg_samples': 1
            })
    
    # Limit number of trials if needed
    if args.max_trials > 0 and len(param_grid) > args.max_trials:
        print(f"Limiting search to {args.max_trials} trials out of {len(param_grid)} possibilities")
        random.shuffle(param_grid)
        param_grid = param_grid[:args.max_trials]
    
    # Store results
    results = []
    
    # Run trials
    for i, hparams in enumerate(param_grid):
        print(f"\n{'='*50}")
        print(f"Trial {i+1}/{len(param_grid)}")
        print(f"Hyperparameters: {hparams}")
        print(f"{'='*50}\n")
        
        # Train model with these hyperparameters
        trial_results = train_model(
            hparams, 
            early_stop=args.early_stop,
            eval_interval=args.eval_interval,
            max_epochs=args.epochs
        )
        
        # Store results
        results.append({
            'hyperparameters': hparams,
            'metrics': trial_results
        })
        
        # Save intermediate results
        with open(args.results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {args.results_file}")
    
    # Find best hyperparameters based on recall@20
    best_recall = -1
    best_config = None
    best_metrics = None
    
    for result in results:
        recall = result['metrics']['valid_recall'][1]  # Recall@20
        if recall > best_recall:
            best_recall = recall
            best_config = result['hyperparameters']
            best_metrics = result['metrics']
    
    print("\n" + "="*50)
    print("Best Hyperparameters:")
    print(json.dumps(best_config, indent=2))
    print("\nBest Metrics:")
    print(f"Valid Recall@20: {best_recall}")
    print("="*50)
    
    return best_config, best_metrics

def standard_training():
    """Run standard training without hyperparameter search"""
    # Setup hyperparameters
    hparams = {
        'lr': eval(args.lr)[0],
        'weight_decay': eval(args.weight_decay)[0],
        'emb_size': eval(args.emb_size)[0],
        'dims': eval(args.dims)[0],
        'steps': eval(args.steps)[0],
        'noise_scale': eval(args.noise_scale)[0],
        'noise_min': eval(args.noise_min)[0],
        'noise_max': eval(args.noise_max)[0],
    }
    
    if args.contrastive:
        hparams.update({
            'temperature': eval(args.temperature)[0],
            'contrastive_weight': eval(args.contrastive_weight)[0],
            'neg_samples': eval(args.neg_samples)[0]
        })
    else:
        hparams.update({
            'temperature': 0.1,
            'contrastive_weight': 0.0,
            'neg_samples': 1
        })
    
    # Train model
    train_model(
        hparams, 
        early_stop=args.early_stop,
        eval_interval=args.eval_interval,
        max_epochs=args.epochs
    )

if __name__ == "__main__":
    if args.search:
        print("Starting hyperparameter search...")
        best_config, best_metrics = hyperparameter_search()
    else:
        print("Running standard training...")
        standard_training()
    
    print("End time: ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))