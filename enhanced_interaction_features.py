import numpy as np
import scipy.sparse as sp
import torch
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import scipy.sparse as sp
import torch
from collections import defaultdict
from tqdm import tqdm
import os
import psutil
import gc

def log_memory_usage(message):
    """Log current memory usage with a message."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024  # in MB
    print(f"Memory usage ({message}): {mem:.2f} MB")

def calculate_ultra_memory_efficient(train_data, batch_size=100, k=100, candidate_pool_size=5000, output_dir=None):
    """
    Ultra memory-efficient calculation of enhanced interaction features.
    
    Args:
        train_data: Sparse matrix of user-item interactions (n_users x n_items)
        batch_size: Number of users to process at once
        k: Number of most similar users to consider for each user
        candidate_pool_size: Number of random users to sample for similarity calculation
        output_dir: Directory to save intermediate results (if None, don't save to disk)
        
    Returns:
        item_counts: Matrix of item interaction counts (n_users x n_items)
        co_interaction_counts: Matrix of co-interaction counts (n_users x n_items)
    """
    print(f"Calculating ultra memory-efficient interaction features...")
    log_memory_usage("start")
    
    n_users, n_items = train_data.shape
    
    # 1. Item interaction counts
    print("Computing item interaction counts...")
    item_counts_vec = np.array(train_data.sum(axis=0)).flatten()
    item_counts = np.tile(item_counts_vec, (n_users, 1))
    log_memory_usage("after item counts")
    
    # 2. Calculate co-interaction counts in batches
    print("Computing co-interaction counts in small batches...")
    
    # Convert to CSR for efficient operations
    train_csr = train_data.tocsr()
    
    # Calculate the number of items each user has interacted with
    user_item_counts = np.array(train_csr.sum(axis=1)).flatten()
    
    # Initialize lists to build CSR matrix directly
    all_data = []
    all_indices = []
    all_indptrs = [0]
    current_ptr = 0
    
    # Process users in batches to reduce memory usage
    num_batches = (n_users + batch_size - 1) // batch_size
    
    # Pre-generate random candidate pools for each batch
    # This limits similarity computation to a smaller set of users
    print("Generating candidate user pools...")
    all_users = np.arange(n_users)
    candidate_pools = {}
    
    for batch_idx in range(num_batches):
        if candidate_pool_size < n_users:
            # Randomly sample users for the candidate pool
            candidate_pools[batch_idx] = np.random.choice(
                all_users, size=min(candidate_pool_size, n_users), replace=False
            )
        else:
            # Use all users if candidate_pool_size >= n_users
            candidate_pools[batch_idx] = all_users
            
    # Process each batch
    for batch_idx in tqdm(range(num_batches), desc="Processing user batches"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_users)
        
        batch_users = list(range(start_idx, end_idx))
        batch_data = []
        batch_indices = []
        batch_indptrs = [0]
        batch_ptr = 0
        
        # Get candidate pool for this batch
        candidate_pool = candidate_pools[batch_idx]
        
        # Process each user in the batch
        for user_idx in batch_users:
            # Get user interactions as sparse vector to save memory
            user_interactions_sparse = train_csr[user_idx]
            
            if user_interactions_sparse.nnz == 0:
                # No interactions for this user
                batch_indptrs.append(batch_ptr)
                continue
            
            # Calculate similarities only with users in the candidate pool
            # This dramatically reduces memory usage
            pool_interactions = train_csr[candidate_pool]
            
            # Get intersection sizes via dot product (sparse operation)
            intersection_sizes = pool_interactions.dot(user_interactions_sparse.T)
            
            # Get user item count
            user_count = user_item_counts[user_idx]
            
            # Get item counts for candidate pool users
            pool_counts = user_item_counts[candidate_pool]
            
            # Calculate union sizes
            union_sizes = user_count + pool_counts - np.array(intersection_sizes.todense()).flatten()
            
            # Calculate Jaccard similarities
            jaccard_similarities = np.zeros(len(candidate_pool))
            nonzero_indices = np.where(union_sizes > 0)[0]
            jaccard_similarities[nonzero_indices] = np.array(
                intersection_sizes[nonzero_indices].todense()
            ).flatten() / union_sizes[nonzero_indices]
            
            # Find top-k similar users from the candidate pool
            if k < len(candidate_pool):
                top_k_pool_indices = np.argsort(jaccard_similarities)[-k:]
                top_k_user_indices = candidate_pool[top_k_pool_indices]
                top_k_similarities = jaccard_similarities[top_k_pool_indices]
            else:
                positive_indices = np.where(jaccard_similarities > 0)[0]
                top_k_user_indices = candidate_pool[positive_indices]
                top_k_similarities = jaccard_similarities[positive_indices]
            
            if len(top_k_user_indices) > 0:
                # Initialize co-counts vector as sparse to save memory
                co_counts = sp.lil_matrix((1, n_items), dtype=np.float32)
                
                # Sum contributions from similar users
                for sim_idx, sim_val in zip(top_k_user_indices, top_k_similarities):
                    # Only contribute if similarity is positive
                    if sim_val > 0:
                        co_counts += sim_val * train_csr[sim_idx]
                
                # Normalize co-counts
                max_val = co_counts.max()
                if max_val > 0:
                    co_counts = co_counts / max_val
                
                # Add to CSR components
                co_counts_coo = co_counts.tocoo()
                
                batch_data.extend(co_counts_coo.data)
                batch_indices.extend(co_counts_coo.col)
                batch_ptr += len(co_counts_coo.data)
            
            batch_indptrs.append(batch_ptr)
            
            # Manually trigger garbage collection periodically
            if (user_idx - start_idx) % 10 == 0:
                gc.collect()
        
        # Save batch results to disk if output_dir is provided
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            batch_file = os.path.join(output_dir, f"batch_{batch_idx}.npz")
            sp.save_npz(
                batch_file, 
                sp.csr_matrix(
                    (batch_data, batch_indices, batch_indptrs),
                    shape=(len(batch_users), n_items)
                )
            )
            print(f"Saved batch {batch_idx} to {batch_file}")
            
            # If saving to disk, we don't need to keep these in memory
            all_data.extend(batch_data)
            all_indices.extend(batch_indices)
            all_indptrs.extend([i + current_ptr for i in batch_indptrs[1:]])
            current_ptr += batch_ptr
        else:
            # Keep results in memory
            all_data.extend(batch_data)
            all_indices.extend(batch_indices)
            all_indptrs.extend([i + current_ptr for i in batch_indptrs[1:]])
            current_ptr += batch_ptr
        
        # Force garbage collection after each batch
        gc.collect()
        log_memory_usage(f"after batch {batch_idx}")
    
    # If we saved to disk, load and combine results
    if output_dir is not None:
        print("Loading batches from disk and combining...")
        co_interaction_counts = sp.csr_matrix((n_users, n_items), dtype=np.float32)
        
        for batch_idx in range(num_batches):
            batch_file = os.path.join(output_dir, f"batch_{batch_idx}.npz")
            batch_matrix = sp.load_npz(batch_file)
            
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_users)
            co_interaction_counts[start_idx:end_idx] = batch_matrix
            
            # Delete file after loading to free disk space
            os.remove(batch_file)
    else:
        # Construct CSR matrix directly from memory
        print("Building final CSR matrix...")
        co_interaction_counts = sp.csr_matrix(
            (all_data, all_indices, all_indptrs),
            shape=(n_users, n_items)
        )
    
    log_memory_usage("final")
    print("Enhanced interaction features calculation complete!")
    return item_counts, co_interaction_counts

def calculate_enhanced_interaction_features_batched_topk(train_data, batch_size=1000, k=100):
    """
    Calculate enhanced interaction features using a batched approach with only top-k similar users.
    Further reduces memory usage by considering only the most similar users.
    
    Args:
        train_data: Sparse matrix of user-item interactions (n_users x n_items)
        batch_size: Number of users to process at once
        k: Number of most similar users to consider for each user
        
    Returns:
        item_counts: Matrix of item interaction counts (n_users x n_items)
        co_interaction_counts: Matrix of co-interaction counts (n_users x n_items)
    """
    print(f"Calculating enhanced interaction features (top-{k} memory-efficient)...")
    n_users, n_items = train_data.shape
    
    # 1. Item interaction counts - vectorized calculation
    print("Computing item interaction counts...")
    item_counts_vec = np.array(train_data.sum(axis=0)).flatten()  # Sum across users
    
    # Create a matrix where each row is the item counts
    item_counts = np.tile(item_counts_vec, (n_users, 1))
    
    # 2. Calculate co-interaction counts in batches
    print("Computing co-interaction counts in batches...")
    
    # Convert to CSR for efficient operations
    train_csr = train_data.tocsr()
    
    # Calculate the number of items each user has interacted with
    user_item_counts = np.array(train_csr.sum(axis=1)).flatten()
    
    # Initialize lists to build CSR matrix directly
    data = []
    indices = []
    indptr = [0]
    current_ptr = 0
    
    # Process users in batches to reduce memory usage
    num_batches = (n_users + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Processing user batches"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_users)
        
        batch_users = range(start_idx, end_idx)
        
        for user_idx in batch_users:
            user_interactions = train_csr[user_idx].toarray().flatten()
            
            if np.sum(user_interactions) == 0:
                # No interactions for this user, add empty row to CSR
                indptr.append(current_ptr)
                continue
                
            # Compute Jaccard similarity with all other users efficiently
            # Get intersection sizes via dot product
            intersection_sizes = train_csr.dot(sp.csr_matrix(user_interactions).T)
            intersection_sizes = intersection_sizes.toarray().flatten()
            
            # Calculate union sizes
            user_count = user_item_counts[user_idx]
            union_sizes = user_count + user_item_counts - intersection_sizes
            
            # Calculate Jaccard similarities
            jaccard_similarities = np.zeros(n_users)
            nonzero_indices = np.where(union_sizes > 0)[0]
            jaccard_similarities[nonzero_indices] = intersection_sizes[nonzero_indices] / union_sizes[nonzero_indices]
            jaccard_similarities[user_idx] = 0  # No self-similarity
            
            # Find top-k similar users
            if k < n_users:
                # Get indices of top-k similarities
                top_k_indices = np.argsort(jaccard_similarities)[-k:]
                # Only keep similarities for top-k users
                top_k_mask = np.zeros(n_users, dtype=bool)
                top_k_mask[top_k_indices] = True
                similar_indices = np.where(np.logical_and(jaccard_similarities > 0, top_k_mask))[0]
            else:
                similar_indices = np.where(jaccard_similarities > 0)[0]
                
            similar_values = jaccard_similarities[similar_indices]
            
            if len(similar_indices) > 0:
                # Compute weighted interactions using matrix operations
                weighted_sum = np.zeros(n_items)
                for sim_idx, sim_val in zip(similar_indices, similar_values):
                    # Get the interactions for this similar user
                    sim_user_interactions = train_csr[sim_idx].toarray().flatten()
                    # Add weighted contribution to the sum
                    weighted_sum += sim_val * sim_user_interactions
                
                # Normalize co-counts
                max_val = np.max(weighted_sum)
                if max_val > 0:
                    co_counts = weighted_sum / max_val
                else:
                    co_counts = weighted_sum
                
                # Add only non-zero elements to CSR components
                nonzero_idx = np.nonzero(co_counts)[0]
                nonzero_vals = co_counts[nonzero_idx]
                
                data.extend(nonzero_vals)
                indices.extend(nonzero_idx)
                current_ptr += len(nonzero_idx)
            
            indptr.append(current_ptr)
    
    # Construct CSR matrix directly
    print("Building final CSR matrix...")
    co_interaction_counts = sp.csr_matrix((data, indices, indptr), shape=(n_users, n_items))
    
    print("Enhanced interaction features calculation complete!")
    return item_counts, co_interaction_counts

def calculate_enhanced_interaction_features_batched(train_data, batch_size=1000):
    """
    Calculate enhanced interaction features using a batched approach to minimize memory usage.
    
    Args:
        train_data: Sparse matrix of user-item interactions (n_users x n_items)
        batch_size: Number of users to process at once
        
    Returns:
        item_counts: Matrix of item interaction counts (n_users x n_items)
        co_interaction_counts: Matrix of co-interaction counts (n_users x n_items)
    """
    print("Calculating enhanced interaction features (memory-efficient)...")
    n_users, n_items = train_data.shape
    
    # 1. Item interaction counts - vectorized calculation
    print("Computing item interaction counts...")
    item_counts_vec = np.array(train_data.sum(axis=0)).flatten()  # Sum across users
    
    # Create a matrix where each row is the item counts
    item_counts = np.tile(item_counts_vec, (n_users, 1))
    
    # 2. Calculate co-interaction counts in batches
    print("Computing co-interaction counts in batches...")
    
    # Convert to CSR for efficient operations
    train_csr = train_data.tocsr()
    
    # Calculate the number of items each user has interacted with
    user_item_counts = np.array(train_csr.sum(axis=1)).flatten()
    
    # Initialize sparse matrix for co-interaction counts
    co_interaction_counts = sp.lil_matrix((n_users, n_items), dtype=np.float32)
    
    # Process users in batches to reduce memory usage
    num_batches = (n_users + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Processing user batches"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_users)
        
        batch_users = range(start_idx, end_idx)
        batch_train = train_csr[batch_users]
        
        # For each user in batch, find similar users and compute co-counts
        for i, user_idx in enumerate(batch_users):
            user_interactions = train_csr[user_idx].toarray().flatten()
            
            if np.sum(user_interactions) == 0:
                continue  # Skip users with no interactions
                
            # Compute Jaccard similarity with all other users efficiently
            # Get intersection sizes via dot product
            intersection_sizes = train_csr.dot(sp.csr_matrix(user_interactions).T)
            intersection_sizes = intersection_sizes.toarray().flatten()
            
            # Calculate union sizes
            user_count = user_item_counts[user_idx]
            union_sizes = user_count + user_item_counts - intersection_sizes
            
            # Calculate Jaccard similarities
            jaccard_similarities = np.zeros(n_users)
            nonzero_indices = np.where(union_sizes > 0)[0]
            jaccard_similarities[nonzero_indices] = intersection_sizes[nonzero_indices] / union_sizes[nonzero_indices]
            jaccard_similarities[user_idx] = 0  # No self-similarity
            
            # Find similar users (use sparse operations)
            similar_indices = np.where(jaccard_similarities > 0)[0]
            similar_values = jaccard_similarities[similar_indices]
            
            if len(similar_indices) > 0:
                # Weighted sum of similar users' interactions
                similar_interactions = train_csr[similar_indices]
                weights = sp.diags(similar_values)
                weighted_interactions = weights.dot(similar_interactions)
                co_counts = np.array(weighted_interactions.sum(axis=0)).flatten()
                
                # Normalize co-counts
                if np.max(co_counts) > 0:
                    co_counts = co_counts / np.max(co_counts)
                
                co_interaction_counts[user_idx] = co_counts
    
    # Convert back to CSR for return
    co_interaction_counts = co_interaction_counts.tocsr()
    
    print("Enhanced interaction features calculation complete!")
    return item_counts, co_interaction_counts

def calculate_enhanced_interaction_features(train_data):
    """
    Calculate enhanced interaction features for a user-item interaction matrix using
    vectorized operations for better performance.
    
    Args:
        train_data: Sparse matrix of user-item interactions (n_users x n_items)
        
    Returns:
        item_counts: Matrix of item interaction counts (n_users x n_items)
        co_interaction_counts: Matrix of co-interaction counts (n_users x n_items)
    """
    print("Calculating enhanced interaction features (vectorized)...")
    n_users, n_items = train_data.shape
    
    # 1. Item interaction counts - vectorized calculation
    print("Computing item interaction counts...")
    item_counts_vec = np.array(train_data.sum(axis=0)).flatten()  # Sum across users
    
    # Create a matrix where each row is the item counts
    item_counts = np.tile(item_counts_vec, (n_users, 1))
    
    # 2. Calculate co-interaction counts using fully vectorized operations
    print("Computing user similarity matrix...")
    # Convert to CSR for efficient operations
    train_csr = train_data.tocsr()
    
    # Compute dot product between users - this gives us the intersection size
    # This is a highly optimized sparse matrix multiplication
    intersection_matrix = train_csr.dot(train_csr.T)
    
    # Calculate the number of items each user has interacted with
    user_item_counts = np.array(train_csr.sum(axis=1)).flatten()
    
    # Vectorized Jaccard similarity calculation
    print("Computing Jaccard similarities...")
    
    # Create matrices for |A| and |B| for each pair
    # For each element (i,j), we need counts for i and j
    user_counts_matrix = np.zeros((n_users, n_users), dtype=np.float32)
    
    # Fill in the user counts (only for users with interactions)
    active_users = np.where(user_item_counts > 0)[0]
    
    for i in tqdm(active_users, desc="Building user similarity matrix"):
        # Set row and column for this user's count
        user_counts_matrix[i, :] = user_item_counts[i]
        user_counts_matrix[:, i] = user_item_counts[i]
    
    # Vectorized calculation for union size: |A| + |B| - |A ∩ B|
    intersection_array = intersection_matrix.toarray()
    union_matrix = user_counts_matrix + user_counts_matrix.T - intersection_array
    
    # Calculate Jaccard similarity
    # Where union is 0, we set it to 1 to avoid division by zero
    union_matrix[union_matrix == 0] = 1
    similarity_matrix = intersection_array / union_matrix
    
    # Set diagonal elements to 0 (no self-similarity)
    np.fill_diagonal(similarity_matrix, 0)
    
    # Convert back to sparse for efficient storage
    user_similarity = sp.csr_matrix(similarity_matrix)
    
    print("Computing co-interaction counts...")
    # Direct matrix multiplication approach:
    # For each user, multiply their similar users by the items those users interacted with
    # This is equivalent to: co_counts = user_similarity * train_data
    co_interaction_counts = user_similarity.dot(train_csr)
    
    # Normalize co-interaction counts (vectorized)
    print("Normalizing co-interaction counts...")
    row_max = co_interaction_counts.max(axis=1).toarray().flatten()
    
    # Create a diagonal matrix with 1/max values to multiply with
    # Replace zeros with 1s to avoid division by zero
    row_max[row_max == 0] = 1
    scaling_diag = sp.diags(1.0 / row_max, 0)
    
    # Apply normalization via matrix multiplication
    co_interaction_counts = scaling_diag.dot(co_interaction_counts)
    
    print("Enhanced interaction features calculation complete!")
    return item_counts, co_interaction_counts

class EnhancedInteractionDataset(torch.utils.data.Dataset):
    """
    Dataset with enhanced interaction features.
    """
    def __init__(self, interaction_tensor, item_counts_tensor, co_interaction_tensor):
        """
        Initialize the dataset.
        
        Args:
            interaction_tensor: Binary user-item interactions (n_users x n_items)
            item_counts_tensor: Item interaction counts (n_users x n_items)
            co_interaction_tensor: Co-interaction counts (n_users x n_items)
        """
        self.interaction_tensor = interaction_tensor
        self.item_counts_tensor = item_counts_tensor
        self.co_interaction_tensor = co_interaction_tensor
        
    def __getitem__(self, index):
        """Get a sample from the dataset."""
        return (
            self.interaction_tensor[index],
            self.item_counts_tensor[index],
            self.co_interaction_tensor[index]
        )
    
    def __len__(self):
        """Get the dataset size."""
        return self.interaction_tensor.shape[0]