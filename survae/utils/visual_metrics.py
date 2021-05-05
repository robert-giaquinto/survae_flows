import os
import time
import numpy as np
import torch

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips


def format_metrics(metrics):
    rval = f"Peak Signal Noise Ratio (PSNR): {metrics['psnr']:0.2f}\nStructural Similarity (SSIM): {metrics['ssim']:0.3f}\nLPIPS: {metrics['lpips']:0.3f}\n"
    return rval


def evaluate_perceptual_quality(model, data_loader, temperature, device):
    evaluate = PerceptualQuality(device=device)
    model.eval()
    with torch.no_grad():
        results = []
        for i, (y, x) in enumerate(data_loader):
            yhat = model.sample(x.to(device), temperature=temperature)

            # pass samples one image at a time
            for yhat_i, y_i in zip(yhat, y):                
                metrics = evaluate.metrics(yhat_i, y_i)
                results.append(metrics)
            
    psnr = np.mean([result['psnr'] for result in results])
    ssim = np.mean([result['ssim'] for result in results])
    lpips = np.mean([result['lpips'] for result in results])
    metrics = {'psnr': psnr,
               'ssim': ssim,
               'lpips': lpips}
    return metrics


class PerceptualQuality():
    def __init__(self, device, net='alex'):
        self.device = device
        self.lpips_model = lpips.LPIPS(net=net).to(self.device)

    def to_HWC(self, img):
        return np.transpose(img, [1, 2, 0])

    def metrics(self, imgA, imgB, as_dict=True):
        if as_dict:
            return {'psnr':  float(self.psnr(imgA, imgB)),
                    'ssim':  float(self.ssim(imgA, imgB)),
                    'lpips': float(self.lpips(imgA, imgB))}
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

        
