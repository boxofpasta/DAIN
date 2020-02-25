import time
import os
from torch.autograd import Variable
import math
import torch

import random
import numpy as np
import numpy
import networks
from my_args import  args
from flow_utils import read_flow_file, pad_image_for_divisibility, read_image
from scipy.misc import imread, imsave
from AverageMeter import  *
import glob
from get_triplet_filepaths import get_triplet_filepaths, get_pair_filepaths

torch.backends.cudnn.benchmark = False # to speed up the

write_depth_image = False

# # Middlebury.
# input_dir = '/scratch/gobi2/tianxingli/raw_data/middlebury-other/other-combined'
# output_dir = '/scratch/gobi2/tianxingli/interp_eval/middlebury/dain'
# triplet_filepaths = get_triplet_filepaths(input_dir, output_dir)

# Hinted val set.
# input_dir = '/h/tianxingli/paper_examples/suppl_examples'
# output_dir = '/scratch/gobi2/tianxingli/interp_eval/hinted_val_set/dain_paper_suppl_examples'
# triplet_filepaths = get_pair_filepaths(input_dir, output_dir)

# # Depth val set.
# input_dir = '/scratch/gobi2/tianxingli/raw_data/depth_val_set'
# output_dir = '/scratch/gobi2/tianxingli/interp_eval/depth_val_set/dain'
# triplet_filepaths = get_pair_filepaths(input_dir, output_dir)
# write_depth_image = True

# # DAVIS-2017 dataset.
# input_dir = '/scratch/gobi2/tianxingli/raw_data/DAVIS_2017_test_dev/JPEGImages/480p'
# output_dir = '/scratch/gobi2/tianxingli/interp_eval/DAVIS_2017/dain'
# triplet_filepaths = get_triplet_filepaths(input_dir, output_dir)

# Creative Flow (hinted).
input_dir = '/scratch/gobi2/tianxingli/raw_data/creative_test_split'
output_dir = '/scratch/gobi2/tianxingli/interp_eval/creative_test_split/dain'
triplet_filepaths = get_triplet_filepaths(input_dir, output_dir)

model = networks.__dict__[args.netName](channel=args.channels,
                            filter_size = args.filter_size,
                            timestep=args.time_step,
                            training=False)

if args.use_cuda:
    model = model.cuda()

args.SAVED_MODEL = './model_weights/best.pth'
if os.path.exists(args.SAVED_MODEL):
    print("The testing model weight is: " + args.SAVED_MODEL)
    if not args.use_cuda:
        pretrained_dict = torch.load(args.SAVED_MODEL, map_location=lambda storage, loc: storage)
        # model.load_state_dict(torch.load(args.SAVED_MODEL, map_location=lambda storage, loc: storage))
    else:
        pretrained_dict = torch.load(args.SAVED_MODEL)
        # model.load_state_dict(torch.load(args.SAVED_MODEL))

    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    # 4. release the pretrained dict for saving memory
    pretrained_dict = []
else:
    print("*****************************************************************")
    print("**** We don't load any trained weights **************************")
    print("*****************************************************************")

model = model.eval() # deploy mode


use_cuda=args.use_cuda
save_which=args.save_which
dtype = args.dtype
unique_id =str(random.randint(0, 100000))
print("The unique id for current testing is: " + str(unique_id))
tot_timer = AverageMeter()
proc_timer = AverageMeter()
end = time.time()
i = 0

torch.set_grad_enabled(False)

for (arguments_strFirst, arguments_strSecond, arguments_strOut) in triplet_filepaths:
    i += 1
    print('Evaluating %d out of %d: %s' % (i, len(triplet_filepaths), arguments_strOut))
    cur_output_dir = os.path.dirname(arguments_strOut)
    if not os.path.exists(cur_output_dir):
        os.makedirs(cur_output_dir, exist_ok=True)

    # fw_flow_file_path = arguments_strFirst[:-4] + '_fw.flo'
    # bw_flow_file_path = arguments_strSecond[:-4] + '_bw.flo'
    fw_flow_file_path = os.path.join(os.path.dirname(arguments_strFirst), 'flow_02.flo')
    bw_flow_file_path = os.path.join(os.path.dirname(arguments_strSecond), 'flow_20.flo')
    flow_0_t = None
    flow_1_t = None
    if os.path.exists(fw_flow_file_path):
        assert os.path.exists(bw_flow_file_path)
        # Pytorch needs [C, H, W].
        flow_0_1 = read_flow_file(fw_flow_file_path)
        flow_1_0 = read_flow_file(bw_flow_file_path)
        flow_0_1 = np.transpose(flow_0_1, axes=[2, 0, 1])
        flow_1_0 = np.transpose(flow_1_0, axes=[2, 0, 1])
        flow_0_t = torch.from_numpy(0.5 * flow_0_1).float().cuda()
        flow_1_t = torch.from_numpy(0.5 * flow_1_0).float().cuda()

    X0 =  torch.from_numpy( np.transpose(read_image(arguments_strFirst), (2,0,1)).astype("float32")/ 255.0).type(dtype)
    X1 =  torch.from_numpy( np.transpose(read_image(arguments_strSecond), (2,0,1)).astype("float32")/ 255.0).type(dtype)
    y_ = torch.FloatTensor()

    assert (X0.size(1) == X1.size(1))
    assert (X0.size(2) == X1.size(2))

    intWidth = X0.size(2)
    intHeight = X0.size(1)
    channel = X0.size(0)
    if not channel == 3:
        continue

    # if intWidth != ((intWidth >> 7) << 7):
    #     # intWidth_pad = ((np.ceil(intWidth >> 7)) << 7)  # more than necessary
    #     intWidth_pad = int(2 ** 7 * np.ceil(float(intWidth) / 2 ** 7))
    #     intPaddingLeft =int(( intWidth_pad - intWidth)/2)
    #     intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
    # else:
    #     intWidth_pad = intWidth
    #     intPaddingLeft = 32
    #     intPaddingRight= 32
    #
    # if intHeight != ((intHeight >> 7) << 7):
    #     # intHeight_pad = ((np.ceil(intHeight >> 7)) << 7)  # more than necessary
    #     intHeight_pad = int(2 ** 7 * np.ceil(float(intHeight) / 2 ** 7))
    #     intPaddingTop = int((intHeight_pad - intHeight) / 2)
    #     intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
    # else:
    #     intHeight_pad = intHeight
    #     intPaddingTop = 32
    #     intPaddingBottom = 32
    if intWidth != ((intWidth >> 7) << 7):
        intWidth_pad = (((intWidth >> 7) + 1) << 7)  # more than necessary
        intPaddingLeft =int(( intWidth_pad - intWidth)/2)
        intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
    else:
        intWidth_pad = intWidth
        intPaddingLeft = 32
        intPaddingRight= 32

    if intHeight != ((intHeight >> 7) << 7):
        intHeight_pad = (((intHeight >> 7) + 1) << 7)  # more than necessary
        intPaddingTop = int((intHeight_pad - intHeight) / 2)
        intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
    else:
        intHeight_pad = intHeight
        intPaddingTop = 32
        intPaddingBottom = 32

    pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom])

    X0 = Variable(torch.unsqueeze(X0,0))
    X1 = Variable(torch.unsqueeze(X1,0))
    X0 = pader(X0)
    X1 = pader(X1)
    if flow_0_t is not None:
        flow_0_t = Variable(torch.unsqueeze(flow_0_t, 0))
        flow_1_t = Variable(torch.unsqueeze(flow_1_t, 0))
        flow_0_t = pader(flow_0_t)
        flow_1_t = pader(flow_1_t)

    if use_cuda:
        X0 = X0.cuda()
        X1 = X1.cuda()
    proc_end = time.time()
    y_s,offset,filter,depth_inv = model(torch.stack((X0, X1),dim = 0), flow_0_t=flow_0_t, flow_1_t=flow_1_t)
    y_ = y_s[save_which]

    proc_timer.update(time.time() -proc_end)
    tot_timer.update(time.time() - end)
    end  = time.time()
    print("*****************current image process time \t " + str(time.time()-proc_end )+"s ******************" )
    if use_cuda:
        X0 = X0.data.cpu().numpy()
        y_ = y_.data.cpu().numpy()
        offset = [offset_i.data.cpu().numpy() for offset_i in offset]
        filter = [filter_i.data.cpu().numpy() for filter_i in filter]  if filter[0] is not None else None
        depth_inv = [depth_inv_i.data.cpu().numpy() for depth_inv_i in depth_inv]
        X1 = X1.data.cpu().numpy()
    else:
        X0 = X0.data.numpy()
        y_ = y_.data.numpy()
        offset = [offset_i.data.numpy() for offset_i in offset]
        filter = [filter_i.data.numpy() for filter_i in filter]
        X1 = X1.data.numpy()

    X0 = np.transpose(255.0 * X0.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))
    y_ = np.transpose(255.0 * y_.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))
    imsave(arguments_strOut, np.round(y_).astype(numpy.uint8))
    print('Writing image to', arguments_strOut)
    if write_depth_image:
        imsave(arguments_strOut[:-4] + '_weights_0.png', depth_inv[0][0][0][intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth])
        imsave(arguments_strOut[:-4] + '_weights_1.png', depth_inv[1][0][0][intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth])

