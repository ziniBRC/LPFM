"""
reference: https://github.com/bonlime/pytorch-tools/blob/master/pytorch_tools/metrics/psnr.py
"""
import os
import pyiqa
import torch
import numpy as np
import argparse
from einops import rearrange
from scipy import stats
import cv2
import torchvision.transforms.functional as F
from sklearn.metrics import normalized_mutual_info_score
# import lpips
# from triton.interpreter.memory_map import torch

class LPIPS:
    def __init__(self, cuda=-1) -> None:
        import lpips
        fn = lpips.LPIPS('vgg')
        self.cuda = cuda
        self.fn = fn.cuda(cuda) if cuda > -1 else fn

    def __call__(self, img0, img1):
        """
        img0 and img1 should be RGB images with range of (0, 255), shape (3, h, w)
        """

        img0 = img0[None]
        img1 = img1[None]
        img0 = img0.permute(0, 3, 1, 2)
        img1 = img1.permute(0, 3, 1, 2)
        img0 = img0.float() / 128
        img1 = img1.float() / 128
        if self.cuda > -1:
            img0 = img0.cuda(self.cuda)
            img1 = img1.cuda(self.cuda)
        return self.fn(img0, img1).item()


class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        img1 = img1.float()
        img2 = img2.float()
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(255.0 / (torch.sqrt(mse) + 1e-8))


class SSIM:
    """Structure Similarity
    img1, img2: [0, 255]"""

    def __init__(self):
        self.name = "SSIM"

    def __call__(self, img1, img2):
        img1 = img1.numpy()
        img2 = img2.numpy()
        if not img1.shape == img2.shape:
            raise ValueError("Input images must have the same dimensions.")
        if img1.ndim == 2:  # Grey or Y-channel image
            return self._ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(self._ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return self._ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError("Wrong input image dimensions.")

    @staticmethod
    def _ssim(img1, img2):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()


def cal_psnr(img1, img2):
    """

    :param img1: 0-255
    :param img2: 0-255
    :return:
    """
    return PSNR()(img1, img2)


def cal_ssim(img1, img2):
    """

    :param img1: 0-255
    :param img2: 0-255
    :return:
    """
    return SSIM()(img1, img2)


def easy_psnr_ssim(img1, img2):
    """

    :param img1: 0-1, n*c*h*w, torch.tensor, float32
    :param img2: same as img1
    :return: psnr and ssim
    """

    img1 = img1.permute([0, 2, 3, 1]) * 255
    img2 = img2.permute([0, 2, 3, 1]) * 255
    psnrs, ssims = [], []
    for i in range(img1.shape[0]):
        p = cal_psnr(img1[i], img2[i])
        s = cal_ssim(img1[i], img2[i])
        psnrs.append(p)
        ssims.append(s)
    return psnrs, ssims


def easy_psnr(img1, img2):
    """

    :param img1: 0-1, n*c*h*w, torch.tensor, float32
    :param img2: same as img1
    :return: psnr and ssim
    """
    img1 = img1.permute([0, 2, 3, 1]) * 255
    img2 = img2.permute([0, 2, 3, 1]) * 255
    p = PSNR()(img1, img2)
    return p


def easy_psnr_ssim_nmi(img1, img2):
    """
    :param img1: 0-1, n*c*h*w, torch.tensor, float32
    :param img2: same as img1
    :return: psnr and ssim
    """

    img1 = img1.permute([0, 2, 3, 1]) * 255
    img2 = img2.permute([0, 2, 3, 1]) * 255
    psnrs, ssims, nmis = [], [], []
    for i in range(img1.shape[0]):
        p = cal_psnr(img1[i], img2[i])
        s = cal_ssim(img1[i], img2[i])
        n = normalized_mutual_info_score(img2[i], img1[i])
        psnrs.append(p)
        ssims.append(s)
        nmis.append(n)
    return psnrs, ssims, nmis


def easy_nmi(predict, gt):
    """
    C*H*W, int (0-255)
    """
    predict = predict.reshape(-1)
    gt = gt.reshape(-1)
    v = normalized_mutual_info_score(predict, gt)
    return v

def mean_95ci(data):
    data = np.array(data)

    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)  # 样本标准差
    n = len(data)
    confidence = 0.95

    # 计算t临界值
    t_critical = stats.t.ppf((1 + confidence) / 2, df=n-1)

    # 计算置信区间
    margin_of_error = t_critical * (std_dev / np.sqrt(n))
    ci = (mean - margin_of_error, mean + margin_of_error)
    return mean, ci[0], ci[1]

    # print(f"均值: {mean:.2f}")
    # print(f"95%置信区间: [{ci[0]:.2f}, {ci[1]:.2f}]")


if __name__ == '__main__':
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    maniqa_metric = pyiqa.create_metric('maniqa', device=device)
    clipiqa_metric = pyiqa.create_metric('clipiqa', device=device)
    musiq_metric = pyiqa.create_metric('musiq', device=device)

    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--gt_dir', default=None, type=str, help='groundtruth files')
    parser.add_argument('--pd_dir', default=None, type=str, help='generation files')
    args = parser.parse_args()

    # save_root = args.results_dir
    # pd_image_root = os.path.join(save_root, 'generations')
    # gt_image_root = os.path.join(save_root, 'inputs')
    pd_image_root = args.pd_dir
    gt_image_root = args.gt_dir
    save_root = os.path.dirname(args.gt_dir)

    psnrs, ssims, lpipss, names = [], [], [], []
    maniqas, clipiqas, musiqs = [], [], []
    psnr_func = PSNR()
    ssim_func = SSIM()
    lpips_func = LPIPS()
    # lpips_func = lpips.LPIPS(net='vgg')

    file_names = os.listdir(pd_image_root)
    for file_name in file_names:
        names.append(file_name)
        pd_image = torch.tensor(cv2.imread(os.path.join(pd_image_root, file_name)))
        gt_image = torch.tensor(cv2.imread(os.path.join(gt_image_root, file_name)))
        if pd_image.shape != gt_image.shape:
            pd_image = rearrange(pd_image, 'h w c -> c h w')
            pd_image = F.resize(pd_image, (gt_image.shape[0], gt_image.shape[1]))
            pd_image = rearrange(pd_image, 'c h w -> h w c')
        # pd_image = pd_image[64:-64, 64:-64, :]
        # gt_image = gt_image[64:-64, 64:-64, :]

        psnrs.append(psnr_func(pd_image, gt_image))
        ssims.append(ssim_func(pd_image, gt_image))
        lpipss.append(lpips_func(pd_image, gt_image))
        pd_image = rearrange(pd_image, 'h w c -> c h w').unsqueeze(0) / 255
        gt_image = rearrange(gt_image, 'h w c -> c h w').unsqueeze(0) / 255
        maniqas.append(maniqa_metric(pd_image, gt_image).item())
        clipiqas.append(clipiqa_metric(pd_image, gt_image).item())
        musiqs.append(musiq_metric(pd_image, gt_image).item())

        print('PSNR: {}, SSIM: {}, LPIPS: {}'.format(psnrs[-1], ssims[-1], lpipss[-1]) )
        print('MANIQA: {}, CLIPIQA: {}, MUSIQA: {}'.format(maniqas[-1], clipiqas[-1], musiqs[-1]) )

    with open(os.path.join(save_root, 'test_metrics.txt'), 'w') as f:
        mean, c1, c2 = mean_95ci(psnrs)
        f.write(f'Mean PSNR: {mean} ({c1}-{c2})\n')
        mean, c1, c2 = mean_95ci(ssims)
        f.write(f'Mean SSIM: {mean} ({c1}-{c2})\n')
        mean, c1, c2 = mean_95ci(lpipss)
        f.write(f'Mean LPIPS: {mean} ({c1}-{c2})\n')
        mean, c1, c2 = mean_95ci(maniqas)
        f.write(f'Mean MANIQA: {mean} ({c1}-{c2})\n')
        mean, c1, c2 = mean_95ci(clipiqas)
        f.write(f'Mean CLIPIQA: {mean} ({c1}-{c2})\n')
        mean, c1, c2 = mean_95ci(musiqs)
        f.write(f'Mean MUSIQA: {mean} ({c1}-{c2})\n')
        f.write('-----------------------------------------------------------\n')
        for i in range(len(names)):
            f.write('Image: {},\tPSNR: {},\tSSIM: {},\tLPIPS: {},'.format(names[i], psnrs[i], ssims[i], lpipss[i]) + \
                    '\tMANIQA: {},\tCLIPIQA: {},\tMUSIQA: {}\n'.format(maniqas[i], clipiqas[i], musiqs[i]))
    #
    # _x1 = x1.permute(2, 0, 1)[None].float() / 255
    # _y1 = y1.permute(2, 0, 1)[None].float() / 255
    # _psnr, _ssim = easy_psnr_ssim(_x1, _y1)
    # print(easy_psnr(_x1, _y1))
    # print('Easy:', _psnr, _ssim)
    # x1 = x1.numpy().astype('uint8')
    # y1 = y1.numpy().astype('uint8')
    # psnr = peak_signal_noise_ratio(x1, y1)
    # print('skimage PSNR:', psnr)
    # ssim = structural_similarity(x1, y1, multichannel=True)
    # print('skimage SSIM:', ssim)
