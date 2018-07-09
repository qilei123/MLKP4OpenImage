# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""

import numpy as np
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps
import PIL
import os
import cPickle


def prepare_roidb(imdb):
    """Enrich the imdb's roidb by adding some derived quantities that
    are useful for training. This function precomputes the maximum
    overlap, taken over ground-truth boxes, between each ROI and
    each ground-truth box. The class with maximum overlap is also
    recorded.
    """
    sizes = []
    #sizes = [PIL.Image.open(imdb.image_path_at(i)).size
    #         for i in xrange(imdb.num_images)]
    
    size_file_name = "sizes.txt"
    
    if os.path.exists(size_file_name):
        sizefile = open(size_file_name)
        
        line = sizefile.readline()
        while line:
            isize = [0,0]
            isize[0] = int(line)
            line = sizefile.readline()
            isize[1] = int(line)
            sizes.append(isize)
            line = sizefile.readline()
    else:
        sizefile = open(size_file_name,'wb')
        for i in xrange(imdb.num_images):
			isize = PIL.Image.open(imdb.image_path_at(i)).size
			sizes.append(isize)
			sizefile.write(str(isize[0])+'\n')
			sizefile.write(str(isize[1])+'\n')
        sizefile.close()
    
    roidb = imdb.roidb
    roidb_file_name = 'Roidb.pkl'
    if os.path.exists(roidb_file_name):
		with open(roidb_file_name,'rb') as fid:
			roidb = cPickle.load(fid)
		print roidb[0]['max_overlaps']
		imdb.roidb = roidb
    else:
		for i in xrange(len(imdb.image_index)):
			roidb[i]['image'] = imdb.image_path_at(i)
			roidb[i]['width'] = sizes[i][0]
			roidb[i]['height'] = sizes[i][1]
			# need gt_overlaps as a dense array for argmax
			gt_overlaps = roidb[i]['gt_overlaps'].toarray()
			# max overlap with gt over classes (columns)
			max_overlaps = gt_overlaps.max(axis=1)
			# gt class that had the max overlap
			max_classes = gt_overlaps.argmax(axis=1)
			roidb[i]['max_classes'] = max_classes
			roidb[i]['max_overlaps'] = max_overlaps
			# sanity checks
			# max overlap of 0 => class should be zero (background)
			zero_inds = np.where(max_overlaps == 0)[0]
			assert all(max_classes[zero_inds] == 0)
			# max overlap > 0 => class should not be zero (must be a fg class)
			nonzero_inds = np.where(max_overlaps > 0)[0]
			assert all(max_classes[nonzero_inds] != 0)
		with open(roidb_file_name, 'wb') as fid:
			cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)			


def add_bbox_regression_targets(roidb):
    """Add information needed to train bounding-box regressors."""
    assert len(roidb) > 0
    assert 'max_classes' in roidb[0], 'Did you call prepare_roidb first?'

    num_images = len(roidb)
    # Infer number of classes from the number of columns in gt_overlaps
    num_reg_classes = 2 if cfg.TRAIN.AGNOSTIC else roidb[0]['gt_overlaps'].shape[1]
    print('num_reg_classes:'+str(num_reg_classes))
    #num_reg_classes = 81
    for im_i in xrange(num_images):
        rois = roidb[im_i]['boxes']
        max_overlaps = roidb[im_i]['max_overlaps']
        max_classes = roidb[im_i]['max_classes']
        roidb[im_i]['bbox_targets'] = \
            _compute_targets(rois, max_overlaps, max_classes)

    if not cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Use fixed / precomputed "means" and "stds" instead of empirical values
        means = np.tile(
            np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (num_reg_classes, 1))
        stds = np.tile(
            np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (num_reg_classes, 1))
    else:
        # Compute values needed for means and stds
        # var(x) = E(x^2) - E(x)^2
        print "compute means and stds"
        means_file = 'means.npy'
        stds_file = 'stds.npy'
        if os.path.exists(means_file) and os.path.exists(stds_file):
            means_array = np.load(means_file)
            means = means_array.tolist()
            stds_array = np.load(stds_file)
            stds = stds_array.tolist()
        else:
			class_counts = np.zeros((num_reg_classes, 1)) + cfg.EPS
			sums = np.zeros((num_reg_classes, 4))
			squared_sums = np.zeros((num_reg_classes, 4))
			for im_i in xrange(num_images):
				targets = roidb[im_i]['bbox_targets']
				for cls in xrange(1, num_reg_classes):
					cls_inds = np.where(targets[:, 0] > 0)[0] if cfg.TRAIN.AGNOSTIC \
						else np.where(targets[:, 0] == cls)[0]
					if cls_inds.size > 0:
						class_counts[cls] += cls_inds.size
						sums[cls, :] += targets[cls_inds, 1:].sum(axis=0)
						squared_sums[cls, :] += \
							(targets[cls_inds, 1:] ** 2).sum(axis=0)

			means = sums / class_counts
			stds = np.sqrt(squared_sums / class_counts - means ** 2)
			np.save(means_file,means)
			np.save(stds_file,stds)

    print 'bbox target means:'
    print means
    print means[1:, :].mean(axis=0)  # ignore bg class
    print 'bbox target stdevs:'
    print stds
    print stds[1:, :].mean(axis=0)  # ignore bg class

	# Normalize targets
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS:
        print "Normalizing targets"
        normalized_roidb_file = "normalizedRoidb.pkl"
        
        if os.path.exists(normalized_roidb_file):
			with open(normalized_roidb_file,'rb') as fid:
				roidb = cPickle.load(fid)
        else:
			for im_i in xrange(num_images):
				targets = roidb[im_i]['bbox_targets']
				for cls in xrange(1, num_reg_classes):
					cls_inds = np.where(targets[:, 0] > 0) if cfg.TRAIN.AGNOSTIC \
						else np.where(targets[:, 0] == cls)[0]
					roidb[im_i]['bbox_targets'][cls_inds, 1:] -= means[cls, :]
					roidb[im_i]['bbox_targets'][cls_inds, 1:] /= stds[cls, :]
			with open(normalized_roidb_file, 'wb') as fid:
				cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)	
        print "Done"
    else:
		print "NOT normalizing targets"
	
    # These values will be needed for making predictions
    # (the predicts will need to be unnormalized and uncentered)
    return means.ravel(), stds.ravel()


def _compute_targets(rois, overlaps, labels):
    """Compute bounding-box regression targets for an image."""
    # Indices of ground-truth ROIs
    gt_inds = np.where(overlaps == 1)[0]
    if len(gt_inds) == 0:
        # Bail if the image has no ground-truth ROIs
        return np.zeros((rois.shape[0], 5), dtype=np.float32)
    # Indices of examples for which we try to make predictions
    ex_inds = np.where(overlaps >= cfg.TRAIN.BBOX_THRESH)[0]

    # Get IoU overlap between each ex ROI and gt ROI
    ex_gt_overlaps = bbox_overlaps(
        np.ascontiguousarray(rois[ex_inds, :], dtype=np.float),
        np.ascontiguousarray(rois[gt_inds, :], dtype=np.float))

    # Find which gt ROI each ex ROI has max overlap with:
    # this will be the ex ROI's gt target
    gt_assignment = ex_gt_overlaps.argmax(axis=1)
    gt_rois = rois[gt_inds[gt_assignment], :]
    ex_rois = rois[ex_inds, :]

    targets = np.zeros((rois.shape[0], 5), dtype=np.float32)
    targets[ex_inds, 0] = labels[ex_inds]
    targets[ex_inds, 1:] = bbox_transform(ex_rois, gt_rois)
    return targets
