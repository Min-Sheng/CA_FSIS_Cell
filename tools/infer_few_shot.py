import os
import cv2
import io
import base64
from scipy.misc import imread
from PIL import Image
from collections import defaultdict, OrderedDict

import torch
from torchvision import transforms
import numpy as np

from tools import _init_paths  # pylint: disable=unused-import
from core.config import cfg, merge_cfg_from_file, merge_cfg_from_list, assert_and_infer_cfg
from core.test_engine import initialize_model_from_cfg
from core.test_few_shot import im_detect_all
import utils.blob as blob_utils
from utils.timer import Timer
import utils.vis as vis_utils
import utils.net as net_utils
import nn as mynn
from modeling import model_builder

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

def initialize(cfg_file, load_ckpt, deform_conv):
    merge_cfg_from_file(cfg_file)
    cfg.MODEL.NUM_CLASSES = 2
    cfg.cuda = True
    cfg.load_ckpt = load_ckpt
    cfg.load_detectron = False
    cfg.VIS_TH = 0.5
    cfg.LOAD_IMAGENET = False
    if deform_conv:
        cfg.USE_DEFORM = True
    else:
        cfg.USE_DEFORM = False
    assert_and_infer_cfg()
    model = initialize_model_from_cfg(cfg, gpu_id=0)
    print("load checkpoint: {}".format(load_ckpt))

    return model

def reload_ckpt(load_ckpt):
    #checkpoint = torch.load(load_ckpt, map_location=lambda storage, loc: storage)
    #net_utils.load_ckpt(model, checkpoint['model'])
    model = model_builder.Generalized_RCNN()
    model.eval()

    model.cuda()
    load_name = load_ckpt
    checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
    net_utils.load_ckpt(model, checkpoint['model'])

    model = mynn.DataParallel(model, cpu_keywords=['im_info', 'roidb'], minibatch=True)

    print("load checkpoint: {}".format(load_ckpt))
    return model

def segment(group_name, target_img_name, query_img_list, model, output_dir, category_name=None):

    def valid_crop(img):
        if img.shape[-1] == 4:
            coords = np.argwhere(img[:,:,3])
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)
            cropped = img[:,:,:3][x_min:x_max+1, y_min:y_max+1]
            return cropped
        else:
            return img

    raw_im = imread(target_img_name)
    ori_im = valid_crop(raw_im)
    if category_name == 'MicroNet':
        im = ori_im[..., 1]
    elif category_name == 'BBBC018_nuclei':
        im = ori_im[..., 2]
    elif category_name == 'BBBC018_cell':
        im = ori_im[..., 0]
    else:
        im = ori_im
    if len(im.shape) == 2:
        im = im[:,:,np.newaxis]
        im = np.concatenate((im,im,im), axis=2)
    if len(ori_im.shape) == 2:
        ori_im = ori_im[:,:,np.newaxis]
        ori_im = np.concatenate((ori_im,ori_im,ori_im), axis=2)

    query = []
    for query_img_name in query_img_list:
        q = imread(query_img_name)
        q = valid_crop(q)
        if category_name == 'MicroNet':
            q = q[..., 1]
        elif category_name == 'BBBC018_nuclei':
            q = q[..., 2]
        elif category_name == 'BBBC018_cell':
            q = q[..., 0]
        if len(q.shape) == 2:
            q = q[:,:,np.newaxis]
            q = np.concatenate((q,q,q), axis=2)
        #q = blob_utils.prep_query_for_blob(q, cfg.PIXEL_MEANS, 64)
        q = blob_utils.pad2square(q, cfg.TRAIN.QUERY_SIZE)
        q, _ = blob_utils.prep_im_for_blob(q, cfg.PIXEL_MEANS, [cfg.TRAIN.QUERY_SIZE],
                        cfg.TRAIN.MAX_SIZE)
        q = blob_utils.im_list_to_blob(q)
        q = torch.from_numpy(q)
        query.append(q)

    timers = defaultdict(Timer)
    catgory = 0
    num_cats = cfg.MODEL.NUM_CLASSES
    box_proposals = None
    
    cls_boxes_i, cls_segms_i, _ = im_detect_all(model, im, [query], catgory, 1, num_cats, box_proposals=box_proposals, timers=timers)
    
    im_det = vis_utils.vis_one_image(
                        ori_im,
                        'det_{:s}'.format(group_name),
                        os.path.join(output_dir, group_name),
                        cls_boxes_i,
                        segms = cls_segms_i,
                        thresh = cfg.VIS_TH,
                        box_alpha = 0.6,
                        show_class = False,
                        save=True,
                        gray_masking=False,
                        draw_bbox =True
                    )
    # return the base64 image (for demo purpose)
    pic_iobytes  = io.BytesIO()
    im_det.save(pic_iobytes, format='PNG')
    byte_data = pic_iobytes.getvalue()
    pic_hash = base64.b64encode(byte_data)

    return pic_hash