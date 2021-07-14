import os
import time
import numpy as np
import torch
import math

import torchvision
from torchvision.transforms.functional import resize
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips


class PerceptualQuality():
    def __init__(self, device, num_bits=8, net='alex', sr_scale_factor=None):
        self.device = device
        self.max_color = 2**num_bits - 1
        self.sr_scale_factor = sr_scale_factor
        self.lpips_model = lpips.LPIPS(net=net).to(self.device)

    def to_HWC(self, img):
        return np.transpose(img, [1, 2, 0])

    def metrics(self, imgA, imgB, as_dict=True):
        if as_dict:
            return {
                'psnr':  float(self.psnr(imgA, imgB)),
                'psnr_check':  float(self.psnr_check(imgA, imgB)),
                'rmse': float(self.rmse(imgA, imgB)),
                'lpips': float(self.lpips(imgA, imgB)),
                'ssim255': float(self.ssim255(imgA, imgB)),
                'ssim':  float(self.ssim(imgA, imgB))
            }
        else:
            return [float(f(imgA, imgB)) for f in [self.psnr, self.ssim, self.lpips]]

    def lpips(self, imgA, imgB):
        #imgA = (imgA / (self.max_color / 2.0)) - 1
        #imgB = (imgB / (self.max_color / 2.0)) - 1
        tA = imgA.to(self.device)
        tB = imgB.to(self.device)
        dist01 = self.lpips_model.forward(tA, tB).item()
        return dist01

    def ssim(self, imgA, imgB):
        # multichannel: If True, treat the last dimension of the array as channels. Similarity calculations are done independently for each channel then averaged.
        imgA = self.to_HWC(imgA.cpu().numpy())
        imgB = self.to_HWC(imgB.cpu().numpy())
        score, diff = ssim(imgA, imgB, full=True, multichannel=True, data_range=self.max_color, gaussian_weights=True, sigma=1.5)
        return score

    def ssim255(self, imgA, imgB):
        imgA = self.to_HWC(imgA.cpu().numpy())
        imgB = self.to_HWC(imgB.cpu().numpy())
        score = ssim(imgA, imgB, data_range=self.max_color, multichannel=True, channel_axis=2, win_size=11)
        return score
    
    def psnr(self, imgA, imgB, shave_border=None):
        imgA = self.to_HWC(imgA.cpu().numpy())
        imgB = self.to_HWC(imgB.cpu().numpy())
        height, width = imgA.shape[:2]

        if shave_border is None:
            shave_border = 0 if self.sr_scale_factor is None else self.sr_scale_factor
        
        imgA = imgA[shave_border:height - shave_border, shave_border:width - shave_border]
        imgB = imgB[shave_border:height - shave_border, shave_border:width - shave_border]
        psnr_val = psnr(imgA, imgB, data_range=self.max_color)
        return psnr_val

    def psnr_check(self, gt, pred):
        #imgA = imgA.cpu().numpy()
        #imgB = imgB.cpu().numpy()
        #return 20 * np.log10(self.max_color) - 10.0 * np.log10(np.mean( (imgA - imgB)**2 ))

        gt = self.to_HWC(gt.cpu().numpy())[:, :, 0]
        pred = self.to_HWC(pred.cpu().numpy())[:, :, 0]


        height, width = pred.shape[:2]
        shave_border = 0 if self.sr_scale_factor is None else self.sr_scale_factor
        pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
        gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
        imdff = pred - gt
        rmse = math.sqrt(np.mean(imdff ** 2))
        if rmse == 0:
            return 100
        
        return 20 * math.log10(self.max_color / rmse)

    def rmse(self, imgA, imgB):
        imgA = imgA.cpu().numpy()
        imgB = imgB.cpu().numpy()
        return math.sqrt(np.mean( (imgA - imgB)**2 ))

    @staticmethod
    def format_metrics(metrics):
        rval = ""
        if 'model_label' in metrics:
            rval += f"Visual Metrics for {metrics['model_label']}:\n"
        rval +=  f"Peak Signal Noise Ratio (PSNR): {metrics['psnr']:0.2f}\n"
        rval += f"Structural Similarity (SSIM): {metrics['ssim']:0.3f}\n"
        rval += f"LPIPS: {metrics['lpips']:0.3f}\n"
        if 'lr_psnr' in metrics:
            rval += f"LR-HR PSNR: {metrics['lr_psnr']:0.2f}\n"
        if 'psnr_check' in metrics:
            rval += f"Manual PSNR: {metrics['psnr_check']:0.2f}\n"
        if 'ssim255' in metrics:
            rval += f"SSIM255: {metrics['ssim255']:0.3f}\n"
        if 'rmse' in metrics:
            rval += f"RMSE: {metrics['rmse']:0.2f}\n"    

        return rval
    
    def evaluate(self, model, data_loader, temperature=None, repeats=1):
        with torch.no_grad():
            results = []
            for i, (y, x) in enumerate(data_loader):
                y = y.to(self.device)
                x = x.to(self.device)

                model_label = model
                if model == "nearest":
                    model_label = "Nearest Neighbor Interpolation"
                    yhat = resize(x, y.shape[-1], interpolation=torchvision.transforms.InterpolationMode.NEAREST)
                elif model == "bicubic":
                    model_label = "Bicubic Interpolation"
                    yhat = resize(x, y.shape[-1], interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
                    #x2 = resize(y, x.shape[-1], interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
                    #yhat = resize(x2, y.shape[-1], interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
                else:
                    model_label = f"PSR Flow (temperature={int(100*temperature)})"
                    assert temperature is not None
                    model.eval()
                    yhat = model.sample(x, temperature=temperature)

                yhat = torch.clamp(yhat, min=0, max=self.max_color)

                # pass samples one image at a time
                for x_i, yhat_i, y_i in zip(x, yhat, y):
                    metrics = self.metrics(y_i, yhat_i)
                    if self.sr_scale_factor is not None:
                        # big_lr = torch.repeat_interleave(torch.repeat_interleave(x_i, self.sr_scale_factor, dim=1), self.sr_scale_factor, dim=2)
                        #metrics['lr_psnr'] = self.psnr(yhat_i, big_lr)
                        metrics['lr_psnr'] = self.psnr(yhat_i[:, ::self.sr_scale_factor, ::self.sr_scale_factor], x_i, shave_border=0)
                    results.append(metrics)

        psnr = np.mean([result['psnr'] for result in results])
        psnr_check = np.mean([result['psnr_check'] for result in results])
        rmse = np.mean([result['rmse'] for result in results])
        ssim = np.mean([result['ssim'] for result in results])
        ssim255 = np.mean([result['ssim255'] for result in results])
        lpips = np.mean([result['lpips'] for result in results])

        if self.sr_scale_factor is not None:
            lr_psnr = np.mean([result['lr_psnr'] for result in results])
            metrics = {'model_label': model_label,
                       'psnr': psnr,
                       'psnr_check': psnr_check,
                       'rmse': rmse,
                       'ssim': ssim,
                       'ssim255': ssim255,
                       'lpips': lpips,
                       'lr_psnr': lr_psnr}
        else:
            metrics = {'psnr': psnr,
                       'ssim': ssim,
                       'lpips': lpips}

        return metrics


        
