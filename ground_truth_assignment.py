# ------------------------------------------------------------------------
# Neural Enhanced Belief Propagation for Multiobject Tracking
# Copyright (c) 2025 MIngchao Liang. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------


import os
import sys
sys.path.append(os.path.dirname(__file__))

from lapsolver import solve_dense

import torch
import torch.nn as nn
import torch.nn.functional as func

def get_ground_truth_affinity(estimations_existing,
                            targets_id_existing,
                            ground_truths_id_existing,
                            estimations_new,
                            targets_id_new,
                            ground_truths):

    batch_size, num_max_gt, dim_gt = ground_truths.shape
    _, num_max_est, _ = estimations_existing.shape
    _, num_max_est_new, _ = estimations_new.shape

    ground_truths_label_new = torch.zeros(batch_size, num_max_est_new, device = estimations_new.device)
    affinity_gt = torch.zeros(batch_size, num_max_est, num_max_est_new, device = estimations_existing.device)

    for batch in range(batch_size):

        dist_threshold = torch.max(ground_truths[batch, :, 3 : 5], dim = -1)[0]
        dist_threshold[dist_threshold > 2] = 2
        dist_threshold[dist_threshold < 0.6] = 0.6

        dist_matrix = torch.sum((estimations_new[batch, :, : 2].unsqueeze(1) - ground_truths[batch, :, : 2].unsqueeze(0) ) ** 2, dim = -1) ** 0.5
        dist_matrix[dist_matrix - dist_threshold[None, :] > 0] = float('nan')

        ground_truths_label_new[batch, :] = torch.any(dist_matrix - 2 < 0, dim = -1)

        row_inds, col_inds = solve_dense(dist_matrix.detach().cpu().numpy())
        ground_truths_id_meas = -torch.ones(num_max_est_new, dtype = torch.long, device = estimations_existing.device)
        ground_truths_id_meas[row_inds] = torch.tensor(col_inds, dtype = torch.long, device = estimations_existing.device)

        ground_truths_id_legacy = ground_truths_id_existing[batch, :]

        affinity_gt[batch, :, :] = torch.logical_and(ground_truths_id_legacy.unsqueeze(1) == ground_truths_id_meas.unsqueeze(0),
                                                     ground_truths_id_legacy.unsqueeze(1) != -1).float()


    affinity_gt_miss = ~torch.any(affinity_gt.bool(), dim = -1, keepdim = True)
    affinity_gt = torch.cat([affinity_gt_miss, affinity_gt], dim = -1)
    affinity_gt_tmp = affinity_gt.clone()

    for batch in range(batch_size):
        dist_threshold = torch.max(estimations_new[batch, :, 3 : 5], dim = -1)[0]
        dist_threshold[dist_threshold > 2] = 2
        dist_threshold[dist_threshold < 0.6] = 0.6

        dist_matrix_self = torch.sum((estimations_existing[batch, :, : 2].unsqueeze(1) - estimations_existing[batch, :, : 2].unsqueeze(0) ) ** 2, dim = -1) ** 0.5
        tmp = torch.logical_and(affinity_gt[batch, :, 0].unsqueeze(1), torch.any(affinity_gt[batch, :, 1:], dim = -1).unsqueeze(0))
        close_targets_inds = torch.any(torch.logical_and(dist_matrix_self < 2, tmp), dim = -1)
        affinity_gt[batch, close_targets_inds, 0] = 0.

    return affinity_gt

def assign_ground_truth_id(estimations_existing,
                           targets_id_existing,
                           ground_truths_id_existing,
                           estimations_new,
                           targets_id_new,
                           ground_truths):

    batch_size, num_max_gt, dim_gt = ground_truths.shape
    _, num_max_est, _ = estimations_existing.shape
    _, num_max_est_new, _ = estimations_new.shape

    ground_truths_id_existing_update = ground_truths_id_existing.clone()
    ground_truths_id_new = -torch.ones(batch_size, num_max_est_new, dtype = torch.long, device = estimations_new.device)
    ground_truths_label_existing = torch.zeros(batch_size, num_max_est, device = estimations_existing.device)
    ground_truths_label_new = torch.zeros(batch_size, num_max_est_new, device = estimations_new.device)
    ground_truths_existing_mask = torch.zeros(batch_size, num_max_gt, dtype = torch.bool)

    ground_truths_id_mask = ground_truths_id_existing != -1

    for batch in range(batch_size):
        ground_truths_id_existing_batch = ground_truths_id_existing[batch, :]
        ground_truths_id_mask_batch = ground_truths_id_mask[batch, :]
        ground_truths_batch = ground_truths[batch, :, :]
        ground_truths_existing_mask_batch = ground_truths_existing_mask[batch, :]

        dist_threshold = torch.max(ground_truths_batch[:, 3 : 5], dim = -1)[0]
        dist_threshold[dist_threshold > 2] = 2
        dist_threshold[dist_threshold < 0.6] = 0.6


        ground_truths_label_existing[batch, ground_truths_id_mask_batch] = \
            (torch.logical_and(~torch.any(torch.isnan(ground_truths_batch[ground_truths_id_existing_batch[ground_truths_id_mask_batch], :]), dim = -1),
                               torch.sum((estimations_existing[batch, ground_truths_id_mask_batch, : 2] -
                                          ground_truths_batch[ground_truths_id_existing_batch[ground_truths_id_mask_batch], : 2]) ** 2, dim = -1) ** 0.5 -
                               dist_threshold[None, ground_truths_id_existing_batch[ground_truths_id_mask_batch]] < 0 )).float()
        ground_truths_existing_mask_batch[ground_truths_id_existing_batch[ground_truths_label_existing[batch, :].bool()]] = True
        

        estimations_existing_batch = estimations_existing[batch, :, :]
        estimations_new_batch = estimations_new[batch, :, :]
        estimations_all_batch = torch.cat([estimations_existing_batch[:, : 2], estimations_new_batch[:, : 2]], dim = -2)
        targets_id_all = torch.cat([targets_id_existing, targets_id_new], dim = -1)

        # -------------------------------------------
        # new POs (legacy POs cannot be reassigned)
        # -------------------------------------------
        # dist_matrix = torch.sum((estimations_new_batch[:, None, : 2] - ground_truths_batch[None, :, : 2]) ** 2, dim = -1) ** 0.5
        # dist_matrix[:, ground_truths_existing_mask_batch] = float('nan')
        # dist_matrix[targets_id_new[batch, :] == -1, :] = float('nan')
        # dist_matrix[dist_matrix - dist_threshold[None, :] > 0] = float('nan')
        # row_inds, col_inds = solve_dense(dist_matrix.detach().cpu().numpy())
        # ground_truths_id_new[batch, row_inds] = torch.tensor(col_inds, dtype = torch.long, device = estimations_new.device)
        # ground_truths_label_new[batch, :] = (ground_truths_id_new[batch, :] != -1).float()

        # mask_tmp = torch.any(ground_truths_id_existing[batch, :, None] == ground_truths_id_new[batch, None, :], dim = -1)
        # ground_truths_id_existing_update[batch, torch.logical_and(~(ground_truths_label_existing[batch, :].bool()), mask_tmp)] = -1

        # -------------------------------------------
        # new POs (legacy POs can be reassigned)
        # -------------------------------------------
        dist_matrix = torch.sum((estimations_all_batch[:, None, : 2] - ground_truths_batch[None, :, : 2]) ** 2, dim = -1) ** 0.5
        dist_matrix[:, ground_truths_existing_mask_batch] = float('nan')
        dist_matrix[torch.cat([ground_truths_label_existing[batch, :].bool(),
                               torch.zeros(num_max_est_new, dtype = torch.bool, device = estimations_new.device)], dim = 0), :] = float('nan')
        dist_matrix[dist_matrix - dist_threshold[None, :] > 0] = float('nan')

        row_inds_existing, col_inds_existing = solve_dense(dist_matrix.detach().cpu().numpy()[: num_max_est, :])

        dist_matrix[:, col_inds_existing] = float('nan')
        row_inds_new, col_inds_new = solve_dense(dist_matrix.detach().cpu().numpy()[num_max_est :, :])

        ground_truths_label_existing[batch, row_inds_existing] = 1
        ground_truths_id_existing_update[batch, ~(ground_truths_label_existing[batch, :].bool())] = -1
        ground_truths_id_existing_update[batch, row_inds_existing] = torch.tensor(col_inds_existing, dtype = torch.long, device = estimations_existing.device)
        ground_truths_id_new[batch, row_inds_new] = torch.tensor(col_inds_new, dtype = torch.long, device = estimations_new.device)

        
        # -------------------------------------------
        # new POs (track initialization)
        # -------------------------------------------
        dist_matrix = torch.sum((estimations_new_batch[:, None, : 2] - ground_truths_batch[None, :, : 2]) ** 2, dim = -1) ** 0.5
        ground_truths_label_new[batch, :] = torch.any(dist_matrix - dist_threshold[None, :] < 0, dim = -1)

        _, counts = torch.unique(torch.cat([ground_truths_id_existing_update[batch, :], ground_truths_id_new[batch, :]], dim = -1), 
                                    sorted=True,
                                    return_counts = True)
        assert torch.all(counts[1 :] == 1)


    return ground_truths_label_existing, ground_truths_label_new, ground_truths_id_existing_update, ground_truths_id_new

