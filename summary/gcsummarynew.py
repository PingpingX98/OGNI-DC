from PIL import Image
from . import BaseSummary, save_ply, PtsUnprojector
import imageio
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import os
import numpy as np
import matplotlib.cm as mcm
import matplotlib.colors as colors
from matplotlib.colors import LogNorm

cmap = 'jet'
cm = plt.get_cmap(cmap)

class OGNIDCSummarynew(BaseSummary):
    def __init__(self, log_dir, mode, args, loss_name, metric_name):
        assert mode in ['train', 'val', 'test'], \
            "mode should be one of ['train', 'val', 'test'] " \
            "but got {}".format(mode)

        super(OGNIDCSummarynew, self).__init__(log_dir, mode, args)

        self.log_dir = log_dir
        self.mode = mode
        self.args = args

        self.loss = []
        self.metric = []

        self.loss_name = loss_name
        self.metric_name = metric_name

        self.path_output = None

        self.t_valid = 0.001

        # ImageNet normalization
        self.img_mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        self.img_std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)

    def update(self, global_step, sample, output):
        if self.loss_name is not None:
            self.loss = np.concatenate(self.loss, axis=0)
            self.loss = np.mean(self.loss, axis=0, keepdims=True)

            msg = [" {:<9s}|  ".format('Loss')]
            for idx, loss_type in enumerate(self.loss_name):
                val = self.loss[0, idx]
                self.add_scalar('Loss/' + loss_type, val, global_step)

                msg += ["{:<s}: {:.4f}  ".format(loss_type, val)]

                if (idx + 1) % 10 == 0:
                    msg += ["\n             "]

            msg = "".join(msg)
            print(msg)

            f_loss = open(self.f_loss, 'a')
            f_loss.write('{:05d} | {}\n'.format(global_step, msg))
            f_loss.close()

        if self.metric_name is not None:
            self.metric = np.concatenate(self.metric, axis=0)
            self.metric = np.mean(self.metric, axis=0, keepdims=True)

            msg = [" {:<9s}|  ".format('Metric')]
            for idx, name in enumerate(self.metric_name):
                val = self.metric[0, idx]
                self.add_scalar('Metric/' + name, val, global_step)

                msg += ["{:<s}: {:.5f}  ".format(name, val)]    # 保留4位小数

                if (idx + 1) % 12 == 0:
                    msg += ["\n             "]

            msg = "".join(msg)
            print(msg)

            f_metric = open(self.f_metric, 'a')
            f_metric.write('{:05d} | {}\n'.format(global_step, msg))
            f_metric.close()

        # Un-normalization
        rgb = sample['rgb'].detach().clone()
        rgb.mul_(self.img_std.type_as(rgb)).add_(self.img_mean.type_as(rgb))
        rgb = rgb.data.cpu().numpy()

        pred = output['pred'].detach().data.cpu().numpy()
        preds = [d.detach().data.cpu().numpy() for d in output['pred_inter']]
        grad_preds = [output['log_depth_grad_init'].detach().data.cpu().numpy()] + [d.detach().data.cpu().numpy() for d in output['log_depth_grad_inter']]

        conf_preds = [d.detach().data.cpu().numpy() for d in output['confidence_depth_grad_inter']]

        dep = sample['dep'].detach().data.cpu().numpy()
        dep_down = output['dep_down'].detach().data.cpu().numpy()
        gt = sample['gt'].detach().data.cpu().numpy()
        mask = (gt > self.t_valid).astype(np.float32)

        conf_input = output['confidence_input'].detach().data.cpu().numpy()

        log_gt = np.log(gt)  # B x 1 x H x W
        log_gt[mask == 0.0] = 0.0

        # compute grad with downsampled gt
        down_rate = self.args.backbone_output_downsample_rate
        if down_rate > 1:
            gt_torch = sample['gt'].detach().data.cpu()
            mask_torch = (gt_torch > self.t_valid).float()
            gt_down = F.avg_pool2d(gt_torch, down_rate)
            mask_down = F.avg_pool2d(mask_torch, down_rate)

            gt_down[mask_down > 0.0] = gt_down[mask_down > 0.0] / mask_down[mask_down > 0.0]
            mask_down[mask_down > 0.0] = 1.0

            gt_down = gt_down.numpy()
            mask_down = mask_down.numpy()

        else:
            gt_down = gt
            mask_down = mask

        log_gt_down = np.log(gt_down)  # B x 1 x H x W
        log_gt_down[mask_down == 0.0] = 0.0

        grad_gt = np.zeros_like(grad_preds[0])
        grad_mask = np.zeros_like(grad_preds[0])

        grad_gt[:, 0, :, 1:] = log_gt_down[:, 0, :, 1:] - log_gt_down[:, 0, :, :-1]
        grad_gt[:, 1, 1:, :] = log_gt_down[:, 0, 1:, :] - log_gt_down[:, 0, :-1, :]

        grad_mask[:, 0, :, 1:] = mask_down[:, 0, :, 1:] * mask_down[:, 0, :, :-1]
        grad_mask[:, 1, 1:, :] = mask_down[:, 0, 1:, :] * mask_down[:, 0, :-1, :]

        grad_gt = grad_gt * grad_mask

        num_summary = rgb.shape[0]
        if num_summary > self.args.num_summary:
            num_summary = self.args.num_summary

        rgb = np.clip(rgb, a_min=0, a_max=1.0)
        dep = np.clip(dep, a_min=0, a_max=self.args.max_depth)
        dep_down = np.clip(dep_down, a_min=0, a_max=self.args.max_depth)
        gt = np.clip(gt, a_min=0, a_max=self.args.max_depth)
        pred = np.clip(pred, a_min=0, a_max=self.args.max_depth)
        preds = [np.clip(item, a_min=0, a_max=self.args.max_depth) for item in preds]
        conf_preds = [np.clip(item, a_min=0, a_max=1.0) for item in conf_preds]
        conf_input = np.clip(conf_input, a_min=0, a_max=1.0)

        list_img = []

        for b in range(0, num_summary):
            rgb_tmp = rgb[b, :, :, :]
            dep_tmp = dep[b, 0, :, :]
            dep_down_tmp = dep_down[b, 0, :, :]
            gt_tmp = gt[b, 0, :, :]
            pred_tmp = pred[b, 0, :, :]
            confidence_x_tmp = [conf[b, 0, :, :] for conf in conf_preds]
            confidence_y_tmp = [conf[b, 1, :, :] for conf in conf_preds]
            conf_input_tmp = conf_input[b, 0, :, :]
            preds_tmp = [d[b, 0, :, :] for d in preds]
            grad_pred_x_tmp = [grad[b, 0, :, :] for grad in grad_preds]
            grad_pred_y_tmp = [grad[b, 1, :, :] for grad in grad_preds]
            grad_gt_x_tmp = grad_gt[b, 0, :, :]
            grad_gt_y_tmp = grad_gt[b, 1, :, :]
            grad_mask_x_tmp = grad_mask[b, 0, :, :]
            grad_mask_y_tmp = grad_mask[b, 1, :, :]
            #error_tmp = depth_err_to_colorbar(pred_tmp, gt_tmp) # H x W x 3
            # normalize for better vis


            depth_normalizer = plt.Normalize(vmin=gt_tmp.min(), vmax=gt_tmp.max())

            grad_pos_x_normalizer = plt.Normalize(vmin=0.0, vmax=max(np.percentile(grad_gt_x_tmp, 95), 0.01))
            grad_neg_x_normalizer = plt.Normalize(vmin=0.0, vmax=max(-np.percentile(grad_gt_x_tmp, 5), 0.01))
            grad_pos_y_normalizer = plt.Normalize(vmin=0.0, vmax=max(np.percentile(grad_gt_y_tmp, 95), 0.01))
            grad_neg_y_normalizer = plt.Normalize(vmin=0.0, vmax=max(-np.percentile(grad_gt_y_tmp, 5), 0.01))

            props = []
            confs = []
            gradxs = []
            gradys = []

            for pred_id in range(len(preds_tmp)):
                pd_tmp = preds_tmp[pred_id]
                # err = np.concatenate([cm(depth_normalizer(pd_tmp))[..., :3], depth_err_to_colorbar(pd_tmp, gt_tmp)], axis=1)
                prop = cm(depth_normalizer(pd_tmp))[..., :3]
                prop = np.transpose(prop[:, :, :3], (2, 0, 1))
                props.append(prop)

            for pred_id in range(len(confidence_x_tmp)):
                confx = confidence_x_tmp[pred_id]
                confy = confidence_y_tmp[pred_id]
                conf = np.concatenate([cm(confx), cm(confy)], axis=1)
                conf = np.transpose(conf[:, :, :3], (2, 0, 1))
                confs.append(conf)

            for pred_id in range(len(grad_pred_x_tmp)):
                # red channel for positive and green channel for negative
                gradx_col = np.zeros_like(grad_gt_x_tmp)[None].repeat(3, 0) # 3 x H x W
                gradx = grad_pred_x_tmp[pred_id]
                grax_pos = grad_pos_x_normalizer(gradx)
                grax_neg = grad_neg_x_normalizer(-gradx)
                gradx_col[0][gradx > 0.0] = grax_pos[gradx > 0.0]
                gradx_col[1][gradx < 0.0] = grax_neg[gradx < 0.0]
                gradxs.append(gradx_col)

            for pred_id in range(len(grad_pred_y_tmp)):
                grady_col = np.zeros_like(grad_gt_y_tmp)[None].repeat(3, 0)  # 3 x H x W
                grady = grad_pred_y_tmp[pred_id]
                gray_pos = grad_pos_y_normalizer(grady)
                gray_neg = grad_neg_y_normalizer(-grady)
                grady_col[0][grady > 0.0] = gray_pos[grady > 0.0]
                grady_col[1][grady < 0.0] = gray_neg[grady < 0.0]
                gradys.append(grady_col)

            props = np.concatenate(props, axis=1)
            confs = np.concatenate(confs, axis=1)
            gradxs = np.concatenate(gradxs, axis=1)
            gradys = np.concatenate(gradys, axis=1)

            dep_tmp = cm(depth_normalizer(dep_tmp))
            dep_down_tmp = cm(depth_normalizer(dep_down_tmp))
            gt_tmp = cm(depth_normalizer(gt_tmp))
            pred_tmp = cm(depth_normalizer(pred_tmp))
            conf_input_tmp = cm(conf_input_tmp)

            dep_tmp = np.transpose(dep_tmp[:, :, :3], (2, 0, 1))
            dep_down_tmp = np.transpose(dep_down_tmp[:, :, :3], (2, 0, 1))
            gt_tmp = np.transpose(gt_tmp[:, :, :3], (2, 0, 1))
            pred_tmp = np.transpose(pred_tmp[:, :, :3], (2, 0, 1))
            #error_tmp = np.transpose(error_tmp[:, :, :3], (2, 0, 1))
            conf_input_tmp = np.transpose(conf_input_tmp[:, :, :3], (2, 0, 1))

            # colorize gt-grad
            # red channel for positive and green channel for negative
            grad_gt_x_col = np.zeros_like(grad_gt_x_tmp)[None].repeat(3, 0)  # 3 x H x W
            gradx_pos = grad_pos_x_normalizer(grad_gt_x_tmp)
            gradx_neg = grad_neg_x_normalizer(-grad_gt_x_tmp)
            grad_gt_x_col[0][grad_gt_x_tmp > 0.0] = gradx_pos[grad_gt_x_tmp > 0.0]
            grad_gt_x_col[1][grad_gt_x_tmp < 0.0] = gradx_neg[grad_gt_x_tmp < 0.0]
            # grad_gt_x_col[2][grad_mask_x_tmp == 0.0] = 0.5 # navy blue for invalid px
            grad_gt_x_tmp = grad_gt_x_col

            grad_gt_y_col = np.zeros_like(grad_gt_y_tmp)[None].repeat(3, 0)  # 3 x H x W
            grady_pos = grad_pos_y_normalizer(grad_gt_y_tmp)
            grady_neg = grad_neg_y_normalizer(-grad_gt_y_tmp)
            grad_gt_y_col[0][grad_gt_y_tmp > 0.0] = grady_pos[grad_gt_y_tmp > 0.0]
            grad_gt_y_col[1][grad_gt_y_tmp < 0.0] = grady_neg[grad_gt_y_tmp < 0.0]
            # grad_gt_y_col[2][grad_mask_y_tmp == 0.0] = 0.5  # navy blue for invalid px
            grad_gt_y_tmp = grad_gt_y_col

            summary_img_name_list = ['rgb', 'sparese_depth', 'sparse_depth_downsampled', 'pred_final', 'gt', 'confidence_input']
            summary_img_list = [rgb_tmp, dep_tmp, dep_down_tmp, pred_tmp, gt_tmp, conf_input_tmp]
            summary_img_name_list.extend(['gradx gt', 'grady gt'])
            summary_img_list.extend([grad_gt_x_tmp, grad_gt_y_tmp])

            summary_img_name_list.extend(['sequence predictions', 'sequence confidence', 'sequence gradx predictions', 'sequence grady predictions'])
            summary_img_list.extend([props, confs, gradxs, gradys])

            list_img.append(summary_img_list)

        for i in range(len(summary_img_name_list)):
            img_tmp = []
            for j in range(len(list_img)):
                img_tmp.append(list_img[j][i])

            img_tmp = np.concatenate(img_tmp, axis=2) # W
            img_tmp = torch.from_numpy(img_tmp)

            self.add_image(self.mode + f'/{summary_img_name_list[i]}', img_tmp, global_step)

        self.flush()

        rmse = self.metric[0, 0]

        # Reset
        self.loss = []
        self.metric = []

        return rmse

    def save(self, epoch, idx, sample, output, id_in_batch=0):
        with torch.no_grad():
            if self.args.save_result_only:
                # self.path_output = '{}/{}/epoch{:04d}'.format(self.log_dir,
                #                                               self.mode, epoch)
                self.path_output = self.log_dir
                os.makedirs(self.path_output, exist_ok=True)
                path_save_pred = os.path.join(self.path_output, 'depth')
                path_save_color = os.path.join(self.path_output, 'depthcolor')
                os.makedirs(path_save_pred, exist_ok=True)
                os.makedirs(path_save_color, exist_ok=True)
                path_save_pred = '{}/{:08d}.png'.format(path_save_pred, idx)
                path_save_color = '{}/{:08d}.png'.format(path_save_color, idx)
                
                pred = output['pred'].detach()

                pred = torch.clamp(pred, min=0)

                pred = pred[0, 0, :, :].data.cpu().numpy()

                pred = (pred*256.0).astype(np.uint16)
                # color_depth = self.ColorizeNew(pred, norm_type='LogNorm', offset=1.)
                color_depth = self.Colorize(pred, min_distance=pred[pred > 0].min(), max_distance=pred.max())
                imageio.imwrite(path_save_pred, pred)
                imageio.imwrite(path_save_color, color_depth)
            else:
                rgb_torch = sample['rgb'].detach()
                dep = sample['dep'].detach()
                pred_torch = output['pred'].detach()
                gt_torch = sample['gt'].detach()
                # K = sample['K'].detach()

                pred_torch = torch.clamp(pred_torch, min=0)

                # Un-normalization
                rgb_torch.mul_(self.img_std.type_as(rgb_torch)).add_(
                    self.img_mean.type_as(rgb_torch))

                rgb = rgb_torch[id_in_batch, :, :, :].data.cpu().numpy()
                dep = dep[id_in_batch, 0, :, :].data.cpu().numpy()
                pred = pred_torch[id_in_batch, 0, :, :].data.cpu().numpy()
                pred_final = pred
                gt = gt_torch[id_in_batch, 0, :, :].data.cpu().numpy()

                # norm = plt.Normalize(vmin=gt.min(), vmax=gt.max())
                # norm = plt.Normalize(vmin=1, vmax=90)

                rgb = np.transpose(rgb, (1, 2, 0))
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
                img_rgb = Image.fromarray(rgb, mode='RGB')
                img_rgb.save(os.path.join(self.path_output, "01_rgb.png"))
                
                self.path_output = '{}/{}/epoch{:04d}/{:08d}'.format(
                    self.log_dir, self.mode, epoch, idx)
                os.makedirs(self.path_output, exist_ok=True)
                
                def save_uint16_gray(array, path):
                    array = np.clip(array, 0, 256.0)
                    array_uint16 = (array * 256).astype(np.uint16)  # float -> 16-bit
                    img_gray = Image.fromarray(array_uint16, mode='I;16')
                    img_gray.save(path)

                # ---------- 保存深度、预测和GT ----------
                save_uint16_gray(dep, os.path.join(self.path_output, "02_depth.png"))
                save_uint16_gray(pred, os.path.join(self.path_output, "03_pred.png"))
                save_uint16_gray(gt, os.path.join(self.path_output, "04_gt.png"))
                

                """ path_save_rgb = '{}/01_rgb.png'.format(self.path_output)
                path_save_dep = '{}/02_dep.png'.format(self.path_output)
                path_save_pred = '{}/03_pred_final.png'.format(self.path_output)
                path_save_pred_visual = '{}/04_pred_final.png'.format(
                    self.path_output)
                    
                pred_uint16 = (pred * 256).astype(np.uint16)

                plt.imsave(path_save_rgb, rgb, cmap=cmap)
                plt.imsave(path_save_dep, cm(norm(dep)))
                imageio.imwrite(path_save_pred, pred_uint16)
                plt.imsave(path_save_pred_visual, cm(norm(pred_final))) """
            

            '''
            if self.args.save_pointcloud_visualization:
                unprojector = PtsUnprojector()
                xyz_gt = unprojector(gt_torch[id_in_batch:id_in_batch+1], K[id_in_batch:id_in_batch+1])  # N x 3
                xyz_pred = unprojector(pred_torch[id_in_batch:id_in_batch + 1], K[id_in_batch:id_in_batch + 1])  # N x 3

                colors = unprojector.apply_mask(rgb_torch[id_in_batch:id_in_batch + 1])

                path_save_pointcloud_gt = '{}/10_pointcloud_gt.ply'.format(self.path_output)
                path_save_pointcloud_pred = '{}/10_pointcloud_pred.ply'.format(self.path_output)

                save_ply(path_save_pointcloud_gt, xyz_gt, colors)
                save_ply(path_save_pointcloud_pred, xyz_pred, colors)
            
            plt.imshow(pred, cmap='gray', vmin=0, vmax=10)
            plt.colorbar()
            plt.show()
            '''
    def ColorizeNew(self, depth, norm_type='LogNorm', offset=1.):
        cmap = mcm.jet
        dm = depth[depth>0].min()
        depth[depth<1e-3] = dm
        depth = (depth - depth.min()) / (depth.max() - depth.min()) + offset
        Norm = getattr(colors, norm_type)
        norm = Norm(vmin=depth.min(), vmax=depth.max(), clip=True)
        m = mcm.ScalarMappable(norm=norm, cmap=cmap)
        depth_color = (255 * m.to_rgba(depth)[:, :, 0:3]).astype(np.uint8)
        return depth_color                   
    
    def log_time(self, epoch, t_total, t_avg,
                 t_bench_avg, t_bench_total):
        """ 
        # 1) TensorBoard
        self.add_scalar('Time/Total_elapsed',   t_total,        epoch)
        self.add_scalar('Time/Avg_traditional', t_avg,          epoch)
        self.add_scalar('Time/Avg_benchmark',   t_bench_avg,    epoch)
        self.add_scalar('Time/Total_benchmark', t_bench_total,  epoch) """

        with open(self.f_metric, 'a') as f:
            f.write(f'{epoch:05d} | '
                    f'Total_elapsed={t_total:.6f}  '
                    f'Avg_traditional={t_avg:.6f}  '
                    f'Avg_benchmark={t_bench_avg:.6f}  '
                    f'Total_benchmark={t_bench_total:.6f}\n')

        self.flush()    

    def Colorize(self, depth, min_distance=None, max_distance=None, radius=None, norm_type='LogNorm', cmap=mcm.jet):
    # norm = colors.Normalize(vmin=min_distance, vmax=max_distance)
    # norm = colors.Normalize(vmin=min_distance, vmax=max_distance,clip=True)
        Norm = getattr(colors, norm_type)
        if min_distance == max_distance:
            min_distance = 1e-5
        if norm_type == 'PowerNorm':
            norm = Norm(vmin=min_distance, vmax=max_distance, clip=True, gamma=0.5)
        else:
            norm = Norm(vmin=min_distance, vmax=max_distance, clip=True)
        # norm = colors.LogNorm(vmin=min_distance, vmax=max_distance,clip=False)
        # norm = colors.PowerNorm(gamma=1. / 2.),
        cmap = cmap
        # cmap = cm.gray
        m = mcm.ScalarMappable(norm=norm, cmap=cmap)
        if radius == None:
            depth_color = (255 * m.to_rgba(depth)[:, :, 0:3]).astype(np.uint8)
            # depth_color[depth <= 0] = [0, 0, 0]
            # depth_color[np.isnan(depth)] = [0, 0, 0]
            # depth_color[depth == np.inf] = [0, 0, 0]
        else:
            pos = np.argwhere(depth > 0)
            depth_color = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
            for i in range(pos.shape[0]):
                color = tuple([int(255 * value) for value in m.to_rgba(depth[pos[i, 0], pos[i, 1]])[0:3]])
                cv2.circle(depth_color, (pos[i, 1], pos[i, 0]), radius, (color[0], color[1], color[2]), -1)
        return depth_color                
    
                
                
                