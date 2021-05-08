import os
import time
import numpy as np
import torch

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips


class PerceptualQuality():
    def __init__(self, device, net='alex'):
        self.device = device
        self.lpips_model = lpips.LPIPS(net=net).to(self.device)

    def to_HWC(self, img):
        return np.transpose(img, [1, 2, 0])

    def metrics(self, imgA, imgB, as_dict=True):
        if as_dict:
            return {
                'psnr':  float(self.psnr(imgA, imgB)),
                'lpips': float(self.lpips(imgA, imgB)),
                'lpips': float(0.0),
                'ssim':  float(self.ssim(imgA, imgB))
            }
        else:
            return [float(f(imgA, imgB)) for f in [self.psnr, self.ssim, self.lpips]]

    def lpips(self, imgA, imgB):
        tA = imgA.to(self.device)
        tB = imgB.to(self.device)
        dist01 = self.lpips_model.forward(tA, tB).item()
        return dist01

    def ssim(self, imgA, imgB):
        # multichannel: If True, treat the last dimension of the array as channels. Similarity calculations are done independently for each channel then averaged.
        imgA = self.to_HWC(imgA.cpu().numpy())
        imgB = self.to_HWC(imgB.cpu().numpy())
        score, diff = ssim(imgA, imgB, full=True, multichannel=True, data_range=255, gaussian_weights=True, sigma=1.5)
        return score

    def psnr(self, imgA, imgB):
        imgA = self.to_HWC(imgA.cpu().numpy())
        imgB = self.to_HWC(imgB.cpu().numpy())
        psnr_val = psnr(imgA, imgB, data_range=255)
        return psnr_val

    def psnr_check(self, imgA, imgB):
        #if imgA.max().item() > 1.0:
        imgA = imgA.cpu().numpy() #/ 255.0
        imgB = imgB.cpu().numpy() #/ 255.0
            
        #return -10.0 * np.log(np.mean(np.square(imgA - imgB))) / np.log(10.0)
        return 20 * np.log10(255.0) - 10.0 * np.log10(np.mean(np.square(imgA - imgB)))

    @staticmethod
    def format_metrics(metrics):
        rval = f"Peak Signal Noise Ratio (PSNR): {metrics['psnr']:0.2f}\nStructural Similarity (SSIM): {metrics['ssim']:0.3f}\nLPIPS: {metrics['lpips']:0.3f}\n"
        if 'lr_psnr' in metrics:
            rval += f"LR-HR PSNR: {metrics['lr_psnr']:0.2f}\n"
        
        return rval
    
    def evaluate(self, model, data_loader, temperature, sr_scale_factor=None):
        model.eval()
        with torch.no_grad():
            results = []
            for i, (y, x) in enumerate(data_loader):
                y = y.to(self.device)
                x = x.to(self.device)
                yhat = model.sample(x, temperature=temperature)

                # pass samples one image at a time
                for x_i, yhat_i, y_i in zip(x, yhat, y):
                    metrics = self.metrics(yhat_i, y_i)
                    if sr_scale_factor is not None:
                        metrics['lr_psnr'] = self.psnr(yhat_i[:, ::sr_scale_factor, ::sr_scale_factor], x_i)
                    results.append(metrics)

        psnr = np.mean([result['psnr'] for result in results])
        ssim = np.mean([result['ssim'] for result in results])
        lpips = np.mean([result['lpips'] for result in results])
        if sr_scale_factor is not None:
            lr_psnr = np.mean([result['lr_psnr'] for result in results])
            metrics = {'psnr': psnr,
                       'ssim': ssim,
                       'lpips': lpips,
                       'lr_psnr': lr_psnr}
        else:
            metrics = {'psnr': psnr,
                       'ssim': ssim,
                       'lpips': lpips}

        return metrics


        
