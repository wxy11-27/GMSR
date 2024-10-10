import torch
import os
import numpy as np
import cv2
import glob
import hdf5storage as hdf5
from utilsfold.utils import reconstruction_patch_image_gpu, save_matv73
import time
from skimage.metrics import structural_similarity
from architecture.GMSR import GMSR
from architecture.MSCNN import Model


# from skimage.measure import  compare_psnr,compare_ssim
if __name__ == "__main__":
    def compute_MRAE(gt, rec):
        gt_hyper = gt
        rec_hyper = rec
        error = np.abs(rec_hyper - gt_hyper) / (gt_hyper+0.001)
        mrae = np.mean(error.reshape(-1))
        return mrae


    def compute_RMSE(gt, rec):
        error = np.power(gt - rec, 2)
        rmse = np.sqrt(np.mean(error))
        return rmse

    def get_reconstruction_gpu(input, model):
        """As the limited GPU memory split the input."""
        model.eval()
        var_input = input.cuda()
        with torch.no_grad():
            start_time = time.time()
            var_output = model(var_input)
            end_time = time.time()

        return end_time-start_time, var_output.cpu()

    def compute_psnr(label,output):

        # assert self.label.ndim == 3 and self.output.ndim == 3

        img_c, img_w, img_h = label.shape
        ref = label.reshape(img_c, -1)
        tar = output.reshape(img_c, -1)
        msr = np.mean((ref - tar) ** 2, 1)
        max1 = np.max(ref, 1)

        psnrall = 10 * np.log10(max1 ** 2 / msr)
        out_mean = np.mean(psnrall)
        print(max1)
        # return out_mean, max1
        return out_mean

    def compute_ergas(self, scale=32):
        d = self.label - self.output
        ergasroot = 0
        for i in range(d.shape[0]):
            ergasroot = ergasroot + np.mean(d[i, :, :] ** 2) / np.mean(self.label[i, :, :]) ** 2
        ergas = (100 / scale) * np.sqrt(ergasroot/(d.shape[0]+1))
        return ergas

    def compute_sam(label,output):
        # assert self.label.ndim == 3 and self.label.shape == self.label.shape

        c, w, h = label.shape
        x_true = label.reshape(c, -1)
        x_pred = output.reshape(c, -1)

        x_pred[np.where((np.linalg.norm(x_pred, 2, 1)) == 0),] += 0.0001

        sam = (x_true * x_pred).sum(axis=1) / (np.linalg.norm(x_true, 2, 1) * np.linalg.norm(x_pred, 2, 1))

        sam = np.arccos(sam) * 180 / np.pi
        # sam = np.arccos(sam)
        mSAM = sam.mean()
        return mSAM

    def compute_ssim(label,output):
        """
        :param x_true:
        :param x_pred:
        :param data_range:
        :param multidimension:
        :return:
        """
        # data_range=1
        # multidimension=False
        mssim =[]
        for i in range(label.shape[0]):
            mssim.append(structural_similarity(label[i, :, :], output[i, :, :],data_range=label.max() - label.min()))

        return np.mean(mssim)


    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model_path = '/share/wangxinying/code/GMSR/results/GMSR/CAVE.pth'
    result_path = '/share/wangxinying/code/GMSR/results/GMSR/'
    img_path = '/share/wangxinying/dataset/Harvard/val_rgb/'
    gt_path = '/share/wangxinying/dataset/Harvard/val_mat/'
    var_name = 'cube'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    model = GMSR(3,31)

    save_point = torch.load(model_path)
    model_param = save_point['state_dict']
    model_dict = {}

    for k1, k2 in zip(model.state_dict(), model_param):
        model_dict[k1] = model_param[k2]
    model.load_state_dict(model_dict)
    model = model.cuda()

    img_path_name = glob.glob(os.path.join(img_path, '*.bmp'))
    img_path_name.sort()
    gt_hyper_name = glob.glob(os.path.join(gt_path, '*.mat'))
    gt_hyper_name.sort()
    mrae = []
    rmse = []
    psnr = []
    ssim = []
    sam = []

    for i in range(len(img_path_name)):
        # load rgb images
        rgb = cv2.imread(img_path_name[i])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = np.float32(rgb) / 255
        # rgb = rgb / rgb.max()
        rgb = np.expand_dims(np.transpose(rgb, [2, 0, 1]), axis=0).copy()
        print(img_path_name[i].split('/')[-1])

        _, img_res = reconstruction_patch_image_gpu(rgb, model,128, 128)
        _, img_res_overlap = reconstruction_patch_image_gpu(rgb[:, :, 128//2:, 128//2:], model,128, 128)
        img_res[128//2:, 128//2:, :] = (img_res[128//2:, 128//2:, :] + img_res_overlap) / 2.0

        rgbFlip = np.flip(rgb, 2).copy()
        _, img_resFlip = reconstruction_patch_image_gpu(rgbFlip, model,  128, 128)
        _, img_res_overlapFlip = reconstruction_patch_image_gpu(rgbFlip[:, :, 128 // 2:, 128 // 2:], model, 128, 128)
        img_resFlip[128 // 2:, 128 // 2:, :] = (img_resFlip[128 // 2:, 128 // 2:, :] + img_res_overlapFlip) / 2.0
        img_resFlip = np.flip(img_resFlip, 0)
        img_res = (img_res + img_resFlip) / 2
        img_res = np.clip(img_res, 0, 1)

        gt_hyper = hdf5.loadmat(gt_hyper_name[i])['ref']
        print(gt_hyper_name[i].split('/')[-1])
        # gt_hyper0 = torch.from_numpy(gt_hyper).unsqueeze(dim=0).permute(0, 3, 1, 2)
        # img_res0 = torch.from_numpy(img_res).unsqueeze(dim=0).permute(0, 3, 1, 2)
        mrae.append(compute_MRAE(img_res, gt_hyper))
        rmse.append(compute_RMSE(img_res, gt_hyper))
        gt_hyper1 = np.transpose(gt_hyper,[2,1,0])
        img_res1 = np.transpose(img_res,[2,1,0])
        psnr.append(compute_psnr(gt_hyper1, img_res1))
        ssim.append(compute_ssim(gt_hyper1, img_res1))
        sam.append(compute_sam(gt_hyper1, img_res1))

        mat_name = img_path_name[i].split('/')[-1][:-4] + '.mat'
        mat_dir = os.path.join(result_path, mat_name)

        save_matv73(mat_dir, var_name, img_res)
        # save_matv73(mat_dir, "bands", bands)
    print('mrae: '+str(sum(mrae)/len(mrae)))
    print('rmse: '+str(sum(rmse)/len(rmse)))
    print('psnr: '+str(sum(psnr)/len(psnr)))
    print('ssim: '+str(sum(ssim)/len(ssim)))
    print('sam: '+str(sum(sam)/len(sam)))
