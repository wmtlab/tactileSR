import torch
import subprocess
import re
import time

def parse_gpu_memory():
    smi_output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used', '--format=csv,noheader,nounits']).decode()
    gpu_info = []
    for line in smi_output.strip().split('\n'):
        index, name, total_mem, used_mem = line.split(', ')
        gpu_info.append({
            'index': int(index),
            'name': name.strip(),
            'total_memory': int(total_mem),
            'used_memory': int(used_mem),
            'free_memory': int(total_mem) - int(used_mem)
        })
    return gpu_info

def select_gpu_with_least_used_memory():
    gpu_info = parse_gpu_memory()
    least_used_gpu = sorted(gpu_info, key=lambda x: x['free_memory'], reverse=True)[0]
    device = "cuda:{}".format(least_used_gpu['index'])
    return least_used_gpu['index'], device, least_used_gpu['name'], least_used_gpu['free_memory']


def test_gpu(device=None, test_time=5, test_memory=1):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    num_elements = int(test_memory * (1024**3) / 4)
    print(f"Allocating a tensor with approximately {num_elements} elements ({test_memory} GB).")

    large_tensor = torch.randn(num_elements, device=device, dtype=torch.float32)
    for _ in range(10):
        large_tensor *= 2.0
        torch.cuda.synchronize()
    
    start_time = time.time()
    elapsed = 0
    while elapsed < test_time:
        large_tensor *= 2.0
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        print(f"Running... Elapsed time: {elapsed:.2f} seconds", end='\r')

    print("Test completed.")


def calculationPSNR(pattern1, pattern2, maxValue, is_printInfo=False):
    """ 
    pattern1: (H, W)
    pattern2: (H, W)
    PSNR = 10 * log_10 (MAX**2 / MSE)
    """
    if is_printInfo:
        print("pattern1:{}, pattern2:{}".format(pattern1.shape, pattern2.shape))
        assert len(pattern1.shape) == len(pattern2),  "pattern1 and pattern2 should be same dimension!"
        assert len(pattern1.shape) == 2, "input should be two dimension!"
        assert pattern1.shape == pattern2.shape, 'pattern should be same shape!' 
    mse  = (pattern1 - pattern2)**2
    mse  = mse.sum() / (pattern1.shape[0]*pattern1.shape[1])
    PSNR = 10 * torch.log10(maxValue**2 / mse)
    return PSNR.data


def calculationSSIM(pattern1, pattern2, C1=0.01**2, C2=0.03**2, is_printInfo=False):
    """ 
    pattern1: (H, W)
    pattern2: (H, W)    
    """
    if is_printInfo:
        print("pattern1:{}, pattern2:{}".format(pattern1.shape, pattern2.shape))
        assert len(pattern1.shape) == len(pattern2),  "pattern1 and pattern2 should be same dimension!"
        assert len(pattern1.shape) == 2, "input should be two dimension!"
        assert pattern1.shape == pattern2.shape, 'pattern should be same shape!' 
    mu1, mu2 = pattern1.mean(), pattern2.mean()
    img1_sq, img2_sq, img12 = pattern1*pattern1, pattern2*pattern2, pattern1*pattern2
    mu1_sq, mu2_sq, mu1_mu2 = mu1*mu1, mu2*mu2, mu1*mu2         # torch & numpy
    sigma1_sq, sigma2_sq, sigma12 = img1_sq.mean() - mu1_sq, img2_sq.mean() - mu2_sq, img12.mean() - mu1_mu2
    ssim = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return ssim


# ssim loss
class SSIM(torch.nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()
        self.channel = 1

    def forward(self, img1, img2):
        return self._ssim(img1, img2)

    def _ssim_structure(self, img1, img2):
        """ 
        img1, img2 channel == 1;
        """
        mu1, mu2 = img1.mean(), img2.mean()
        img1_sq, img2_sq, img12 = img1*img1, img2*img2, img1*img2
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1*mu2
        sigma1_sq, sigma2_sq, sigma12 = img1_sq.mean() - mu1_sq, img2_sq.mean() - mu2_sq, img12.mean() - mu1_mu2
        C3 = 0.03**2
        img_structure = (sigma12+C3)/(sigma1_sq*sigma2_sq+C3)
        return img_structure
            
    def _ssim(self, img1, img2):
        mu1, mu2 = img1.mean(), img2.mean()
        img1_sq, img2_sq, img12 = img1*img1, img2*img2, img1*img2
        # mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1*mu2
        mu1_sq, mu2_sq, mu1_mu2 = mu1*mu1, mu2*mu2, mu1*mu2
        sigma1_sq, sigma2_sq, sigma12 = img1_sq.mean() - mu1_sq, img2_sq.mean() - mu2_sq, img12.mean() - mu1_mu2
        C1 = 0.01**2
        C2 = 0.03**2
        ssim = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        return ssim

if __name__ == "__main__":
    gpu_index, gpu_name, free_memory = select_gpu_with_least_used_memory()
    print(f"选择的GPU是 {gpu_index} - {gpu_name}，可用显存 {free_memory} MB")
    # device = "cuda:{}".format(gpu_index) if torch.cuda.is_available() else "cpu"
    device = "cuda:{}".format(gpu_index)
    test_gpu(device, test_time=5, test_memory=1)