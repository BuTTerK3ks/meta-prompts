# ------------------------------------------------------------------------------
# The code is from GLPDepth (https://github.com/vinvino02/GLPDepth).
# For non-commercial purpose only (research, evaluation etc).
# -----------------------------------------------------------------------------

import os
import cv2
import numpy as np
from datetime import datetime

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt


from models_depth.model import MetaPromptDepth
from models_depth.optimizer import build_optimizers
import utils_depth.metrics as metrics
from utils_depth.criterion import SiLogLoss
import utils_depth.logging as logging

from dataset.base_dataset import get_dataset
from configs.train_options import TrainOptions
import glob
import utils
from tqdm import tqdm

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from wecreateyour.load_dataset import ThreeDCDataset

metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog']


def load_model(ckpt, model, optimizer=None):
    ckpt_dict = torch.load(ckpt, map_location='cpu')
    # keep backward compatibility
    if 'model' not in ckpt_dict and 'optimizer' not in ckpt_dict:
        state_dict = ckpt_dict
    else:
        state_dict = ckpt_dict['model']
    weights = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            weights[key[len('module.'):]] = value
        else:
            weights[key] = value

    model.load_state_dict(weights)

    if optimizer is not None:
        optimizer_state = ckpt_dict['optimizer']
        optimizer.load_state_dict(optimizer_state)


def main():
    opt = TrainOptions()
    args = opt.initialize().parse_args()
    print(args)

    #TODO Self set
    args.rank = 0
    args.gpu = "cuda"
    args.shift_window_test=False
    args.pro_bar=True

    #TODO Changed distributed mode here
    #utils.init_distributed_mode_simple(args)
    print(args)
    device = torch.device(args.gpu)


    pretrain = args.pretrained.split('.')[0]
    maxlrstr = str(args.max_lr).replace('.', '')
    minlrstr = str(args.min_lr).replace('.', '')
    layer_decaystr = str(args.layer_decay).replace('.', '')
    weight_decaystr = str(args.weight_decay).replace('.', '')
    num_filter = str(args.num_filters[0]) if args.num_deconv > 0 else ''
    num_kernel = str(args.deconv_kernels[0]) if args.num_deconv > 0 else ''
    name = [args.dataset, str(args.batch_size), pretrain.split('/')[-1], 'deconv'+str(args.num_deconv), \
        str(num_filter), str(num_kernel), str(args.crop_h), str(args.crop_w), maxlrstr, minlrstr, \
        layer_decaystr, weight_decaystr, str(args.epochs)]
    if args.exp_name != '':
        name.append(args.exp_name)

    exp_name = '_'.join(name)
    print('This experiments: ', exp_name)

    # Logging
    if args.rank == 0:
        exp_name = '%s_%s' % (datetime.now().strftime('%m%d'), exp_name)
        log_dir = os.path.join(args.log_dir, exp_name)
        logging.check_and_make_dirs(log_dir)
        writer = SummaryWriter(logdir=log_dir)
        log_txt = os.path.join(log_dir, 'logs.txt')  
        logging.log_args_to_txt(log_txt, args)

        global result_dir
        result_dir = os.path.join(log_dir, 'results')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
    else:
        log_txt = None
        log_dir = None
        
    model = MetaPromptDepth(args=args)

    # CPU-GPU agnostic settings
    
    cudnn.benchmark = True
    model.to(device)
    model_without_ddp = model
    #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)


    #TODO Replaced dataset loading

    # Function to load IDs from a text file
    def load_ids_from_file(file_path):
        with open(file_path, 'r') as file:
            ids = [line.strip() for line in file]
        return ids

    dataset_path = '/home/grannemann/Allgemein/Christian/LOOXIS/wecreateyour'

    # Load training and validation IDs
    train_ids = load_ids_from_file(os.path.join(dataset_path, 'face_detection', 'train.txt'))
    val_ids = load_ids_from_file(os.path.join(dataset_path, 'face_detection', 'val.txt'))

    resize_size = (512, 512)

    # Initialize datasets with specific splits
    train_dataset = ThreeDCDataset(data_path=dataset_path, ids=train_ids, resize_size=resize_size)
    val_dataset = ThreeDCDataset(data_path=dataset_path, ids=val_ids, resize_size=resize_size)

    # Creating PyTorch data loaders for the train and validation sets
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, prefetch_factor=5)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, prefetch_factor=5)

    # Training settings
    criterion_d = SiLogLoss()

    optimizer = build_optimizers(model, dict(type='AdamW', lr=args.max_lr, betas=(0.9, 0.999), weight_decay=args.weight_decay,
                constructor='LDMOptimizerConstructor',
                paramwise_cfg=dict(layer_decay_rate=args.layer_decay, no_decay_names=['relative_position_bias_table', 'rpe_mlp', 'logit_scale'])))

    start_ep = 1
    if args.resume_from:
        load_model(args.resume_from, model.module, optimizer)
        strlength = len('_model.ckpt')
        resume_ep = int(args.resume_from[-strlength-2:-strlength])
        print(f'resumed from epoch {resume_ep}, ckpt {args.resume_from}')
        start_ep = resume_ep + 1
    if args.auto_resume:
        ckpt_list = glob.glob(f'{log_dir}/epoch_*_model.ckpt')
        strlength = len('_model.ckpt')
        idx = [ckpt[-strlength-2:-strlength] for ckpt in ckpt_list]
        if len(idx) > 0:
            idx.sort(key=lambda x: -int(x))
            ckpt = f'{log_dir}/epoch_{idx[0]}_model.ckpt'
            load_model(ckpt, model.module, optimizer)
            resume_ep = int(idx[0])
            print(f'resumed from epoch {resume_ep}, ckpt {ckpt}')
            start_ep = resume_ep + 1

    global global_step
    iterations = len(train_loader)
    global_step = iterations * (start_ep - 1)

    best_rmse = 1000

    # Perform experiment
    for epoch in range(start_ep, args.epochs + 1):
        print('\nEpoch: %03d - %03d' % (epoch, args.epochs))
        loss_train = train(train_loader, model, criterion_d, log_txt, optimizer=optimizer, 
                           device=device, epoch=epoch, args=args)
        if args.rank == 0:
            writer.add_scalar('Training loss', loss_train, epoch)

        if epoch % args.val_freq == 0:
            results_dict, loss_val = validate(val_loader, model, criterion_d, 
                                              device=device, epoch=epoch, args=args)
            if args.rank == 0:
                writer.add_scalar('Val loss', loss_val, epoch)

                result_lines = logging.display_result(results_dict)
                if args.kitti_crop:
                    print("\nCrop Method: ", args.kitti_crop)
                print(result_lines)

                with open(log_txt, 'a') as txtfile:
                    txtfile.write('\nEpoch: %03d - %03d' % (epoch, args.epochs))
                    txtfile.write(result_lines)                

                for each_metric, each_results in results_dict.items():
                    writer.add_scalar(each_metric, each_results, epoch)

        if args.rank == 0:
            if args.save_model:
                torch.save(
                    {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict()
                    },
                    os.path.join(log_dir, 'last.ckpt'))
            
            if results_dict['rmse'] < best_rmse:
                best_rmse = results_dict['rmse']
                torch.save(
                    {
                        'model': model_without_ddp.state_dict(),
                    },
                    os.path.join(log_dir, 'best.ckpt'))

def visualize_image(input_RGB, index=0):
    """
    Visualize an image from a batch.

    Parameters:
    - input_RGB: a batch of images as a PyTorch tensor.
    - index: index of the image in the batch to visualize.
    """
    if input_RGB.dim() == 4:  # Check if the input tensor is in the format [B, C, H, W]
        image = input_RGB[index]  # Select the image at the specified index
    else:
        raise ValueError("Input tensor is not in the expected [B, C, H, W] format")

    # Convert the image from PyTorch tensor to a NumPy array and change the channel order from [C, H, W] to [H, W, C]
    image = image.cpu().numpy().transpose(1, 2, 0)

    # If the image data is normalized, you might need to denormalize it (e.g., multiply by the standard deviation and add the mean for each channel)
    # For simplicity, this step is not included here.

    # Display the image
    plt.imshow(image)
    plt.axis('off')  # Turn off axis labels and ticks
    plt.show()

def train(train_loader, model, criterion_d, log_txt, optimizer, device, epoch, args):
    global global_step
    model.train()
    if args.rank == 0:
        depth_loss = logging.AverageMeter()
    half_epoch = args.epochs // 2
    iterations = len(train_loader)
    result_lines = []

    # Wrap the training loader with tqdm for a progress bar
    train_loader_tqdm = tqdm(enumerate(train_loader), total=iterations, desc=f"Epoch {epoch}/{args.epochs}")

    for batch_idx, batch in train_loader_tqdm:
        global_step += 1

        if global_step < iterations * half_epoch:
            current_lr = (args.max_lr - args.min_lr) * (global_step / iterations / half_epoch) ** 0.9 + args.min_lr
        else:
            current_lr = max(args.min_lr, (args.min_lr - args.max_lr) * (
                        global_step / iterations / half_epoch - 1) ** 0.9 + args.max_lr)

        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr * param_group['lr_scale']

        input_RGB = batch['image'].to(device)

        #visualize_image(input_RGB, index=0)

        depth_gt = batch['depth'].to(device)
        #print(np.max(batch['depth'].numpy()[0]))
        mask = batch['mask'].to(device)
        class_ids = None
        if 'class_id' in batch:
            class_ids = batch['class_id']
        preds = model(input_RGB)

        optimizer.zero_grad()
        pred_value = list(preds.values())
        loss_d = 0
        # Iterate over your predictions if they are in a list or dict
        for pred in pred_value:
            # Apply the mask: element-wise multiplication of the mask with the loss map
            # The mask should be 1 for relevant pixels and 0 for irrelevant ones
            pred = pred.squeeze(dim=1) * mask
            depth_gt = depth_gt * mask
            unmasked_loss = criterion_d(pred, depth_gt)
            masked_loss = unmasked_loss * mask  # Apply mask here
            loss_d += masked_loss.sum()
        # Normalize the loss by the number of predictions and by the sum of the mask values
        # to account for the actual number of relevant pixels
        loss_d = loss_d / len(pred_value) / mask.sum()
        # loss_d = criterion_d(preds['pred_d'].squeeze(dim=1), depth_gt)

        if args.rank == 0:
            depth_loss.update(loss_d.item(), input_RGB.size(0))

        loss_d.backward()
        optimizer.step()

        if args.rank == 0:
            # Update the progress bar
            train_loader_tqdm.set_postfix(
                {'Depth Loss': f'{depth_loss.val:.4f} ({depth_loss.avg:.4f})', 'LR': f'{current_lr:.6f}'})

            if batch_idx % args.print_freq == 0:
                result_line = 'Epoch: [{0}][{1}/{2}]\t' \
                              'Loss: {loss}, LR: {lr}\n'.format(
                    epoch, batch_idx, iterations,
                    loss=depth_loss.avg, lr=current_lr
                )
                result_lines.append(result_line)
                print(result_line)

    if args.rank == 0:
        with open(log_txt, 'a') as txtfile:
            txtfile.write('\nEpoch: %03d - %03d' % (epoch, args.epochs))
            for result_line in result_lines:
                txtfile.write(result_line)

    return loss_d


def validate(val_loader, model, criterion_d, device, epoch, args):

    if args.rank == 0:
        depth_loss = logging.AverageMeter()
    model.eval()

    ddp_logger = utils.MetricLogger()

    result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0

    for batch_idx, batch in enumerate(val_loader):
        input_RGB = batch['image'].to(device)
        depth_gt = batch['depth'].to(device)
        mask = batch['mask'].to(device)
        filename = batch['filename'][0]
        class_id = None
        if 'class_id' in batch:
            class_id = batch['class_id']

        with torch.no_grad():
            if args.shift_window_test:
                bs, _, h, w = input_RGB.shape
                assert w > h and bs == 1
                interval_all = w - h
                interval = interval_all // (args.shift_size-1)
                sliding_images = []
                sliding_masks = torch.zeros((bs, 1, h, w), device=input_RGB.device)
                class_ids = []
                for i in range(args.shift_size):
                    sliding_images.append(input_RGB[..., :, i*interval:i*interval+h])
                    sliding_masks[..., :, i*interval:i*interval+h] += 1
                    class_ids.append(class_id)
                input_RGB = torch.cat(sliding_images, dim=0)
                if class_id is not None:
                    class_ids = torch.cat(class_ids, dim=0)
            if args.flip_test:
                input_RGB = torch.cat((input_RGB, torch.flip(input_RGB, [3])), dim=0)
                if class_id is not None:
                    class_ids = torch.cat((class_ids, class_ids), dim=0)
            pred = model(input_RGB)
        pred_value = list(pred.values())
        pred_d = pred_value[-1]
        # pred_d = pred['pred_d']
        if args.flip_test:
            batch_s = pred_d.shape[0]//2
            pred_d = (pred_d[:batch_s] + torch.flip(pred_d[batch_s:], [3]))/2.0
        if args.shift_window_test:
            pred_s = torch.zeros((bs, 1, h, w), device=pred_d.device)
            for i in range(args.shift_size):
                pred_s[..., :, i*interval:i*interval+h] += pred_d[i:i+1]
            pred_d = pred_s/sliding_masks

        pred_d = pred_d * mask
        depth_gt = depth_gt * mask

        pred_d = pred_d.squeeze()
        depth_gt = depth_gt.squeeze()



        unmasked_loss = criterion_d(pred_d.squeeze(), depth_gt)
        masked_loss = unmasked_loss * mask  # Apply mask here
        masked_loss = masked_loss.sum()
        loss_d = masked_loss / len(pred_value) / mask.sum()



        ddp_logger.update(loss_d=loss_d.item())

        if args.rank == 0:
            depth_loss.update(loss_d.item(), input_RGB.size(0))

        pred_crop, gt_crop = metrics.cropping_img(args, pred_d, depth_gt)
        computed_result = metrics.eval_depth(pred_crop, gt_crop)

        if args.rank == 0:
            save_path = os.path.join(result_dir, filename)
            save_path = save_path + '.npy'  # Ensuring the file is saved with .npy extension

            if args.save_result:
                # Convert the tensor to a NumPy array
                pred_d_numpy = pred_d.cpu().numpy()

                pred_d_numpy = pred_d_numpy * 100

                # Save the NumPy array to a .npy file
                np.save(save_path, pred_d_numpy)
                    
        if args.rank == 0:
            loss_d = depth_loss.avg
            if args.pro_bar:
                logging.progress_bar(batch_idx, len(val_loader), args.epochs, epoch)

        ddp_logger.update(**computed_result)
        for key in result_metrics.keys():
            result_metrics[key] += computed_result[key]

    # for key in result_metrics.keys():
    #     result_metrics[key] = result_metrics[key] / (batch_idx + 1)

    ddp_logger.synchronize_between_processes()

    for key in result_metrics.keys():
        result_metrics[key] = ddp_logger.meters[key].global_avg

    loss_d = ddp_logger.meters['loss_d'].global_avg

    return result_metrics, loss_d


if __name__ == '__main__':
    main()
