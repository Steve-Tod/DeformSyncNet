import torch
import numpy as np
from tqdm import tqdm


def iterate_in_chunks(l, n):
    """Yield successive 'n'-sized chunks from iterable 'l'.
    Note: last chunk will be smaller than l if n doesn't divide l perfectly.
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def minimum_matching_distance(ref_pcs, sample_pcs, batch_size, structural_dist, verbose=False):
    """ Computes the MMD between two sets of point-clouds.
    :param ref_pcs: (M x N x 3) set of M point-clouds that is considered as the ground-truth (target set).
    :param sample_pcs: (K x N x 3) set of K point-clouds that is supposed to be close under the MMD to the target.
    :param batch_size: The batch-size over which we will consider elements among the two sets
        (bigger is faster, but be minded of quadratic memory footprint).
    :param structural_dist: (function) takes two equally size-sets of point clouds and returns the matching distances
        e.g. Chamfer or EMD.
    :param verbose: (boolean, default False), if true show progress.
    :return: mmd (float), all the matching distances (K sized-list),
        the matched elements/ids pointing to elements of the sample_pcs (K-sized list)
    """

    n_ref, n_pc_points, pc_dim = ref_pcs.shape
    _, n_pc_points_s, pc_dim_s = sample_pcs.shape
    
    if n_pc_points != n_pc_points_s or pc_dim != pc_dim_s:
        raise ValueError('Incompatible size of point-clouds.')
            
    matched_ref_items = list()  # for each sample who is the (minimal-distance) gt pc
    matched_dists = list()  # for each sample what is the (minimal-distance) from the match gt pc
    
    all_refs_iter = iter(range(n_ref))    
    for i in (tqdm(all_refs_iter) if verbose else all_refs_iter):
        min_in_all_batches = list()
        loc_in_all_batches = list()
        for sample_chunk in iterate_in_chunks(sample_pcs, batch_size):            
            n_samples = len(sample_chunk)
            ref_i = ref_pcs[i].repeat(n_samples, 1, 1)
            all_dist_in_batch = structural_dist(ref_i, sample_chunk)            
            location_of_min = all_dist_in_batch.argmin()
            min_in_batch = all_dist_in_batch[location_of_min]  # Best distance, of in-batch samples matched to single ref pc.
            
            min_in_all_batches.append(min_in_batch)
            loc_in_all_batches.append(location_of_min)
                        
        min_in_all_batches = torch.stack(min_in_all_batches)        
        min_batch_for_ref_i = torch.argmin(min_in_all_batches)
        min_dist_for_ref_i = min_in_all_batches[min_batch_for_ref_i]
        
        min_loc_inside_min_batch = torch.stack(loc_in_all_batches)[min_batch_for_ref_i]        
        matched_item_for_ref_i = min_batch_for_ref_i * batch_size + min_loc_inside_min_batch  
        
        matched_dists.append(min_dist_for_ref_i)
        matched_ref_items.append(matched_item_for_ref_i)

    matched_dists = torch.stack(matched_dists)
    matched_ref_items = torch.stack(matched_ref_items)    
    mmd = torch.mean(matched_dists).item()
    
    return mmd, matched_dists.cpu().numpy(), matched_ref_items.cpu().numpy()


def coverage(ref_pcs, sample_pcs, batch_size, structural_dist, verbose=False):
    """ Computes the Coverage between two sets of point-clouds.
    :param ref_pcs: (M x N x 3) set of M point-clouds that is considered as the ground-truth (target set).
    :param sample_pcs: (K x N x 3) set of K point-clouds that is supposed to 'cover' the target set.
    :param batch_size: The batch-size over which we will consider elements among the two sets
        (bigger is faster, but be minded of quadratic memory footprint).
    :param structural_dist: (function) takes two equally size-sets of point clouds and returns the matching distances
        e.g. Chamfer or EMD.
    :param verbose: (boolean, default False), if true show progress.
    :return: coverage (float), the matched elements/ids pointing to elements of the ref_pcs (M-sized list)
    """
    matched_elements = minimum_matching_distance(sample_pcs, ref_pcs, batch_size, structural_dist, verbose=verbose)[-1]
    return len(np.unique(matched_elements)) / len(ref_pcs), matched_elements
