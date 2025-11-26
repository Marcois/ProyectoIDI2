import torch
from torch.utils.data import Dataset
import numpy as np
import io
import cv2
from scipy.signal import wiener
from torchvision import transforms
from PIL import Image, ImageChops, ImageEnhance

class ImageDataset(Dataset):

    def __init__(self, df, transform=None, return_prnu=False, 
                return_ela=False, ela_quality=90, img_size=256):
        """
        df: Dataframe with image data ["file_name", "label"]
        transform: torchvision transforms
        return_prnu: True -> return (RGB+PRNU) 4-channel tensor
        return_ela: True -> return ELA image
        ela_quality: JPEG quality for recompression
        img_size: Size of image for rescaling
        """
        self.df = df
        self.transform = transform if transform else transforms.ToTensor()
        self.return_prnu = return_prnu
        self.return_ela = return_ela
        self.ela_quality = ela_quality
        self.img_size = img_size

    def __extract_prnu__(self,rgb):
        """
        rgb: rgb image
        Uses safe PRNU extraction (wavelet denoise -> noise residual).
        """
        gray = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2GRAY).astype(np.float32)
        denoise = cv2.fastNlMeansDenoising(gray.astype(np.uint8), None, h=5, templateWindowSize=3, searchWindowSize=13)
        noise = gray - denoise.astype(np.float32)
        noise = noise - cv2.GaussianBlur(noise, (21,21), 0)
        noise = wiener(noise, (5,5))
        prnu = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        return prnu

    def __extract_ela__(self, rgb):
        """
        rgb: rgb image
        returns: 1-channel ELA image (float32 0-1)
        """
        buffer = io.BytesIO()
        rgb.save(buffer, format="JPEG", quality=self.ela_quality)
        buffer.seek(0)
        recompressed = Image.open(buffer)

        difference = ImageChops.difference(rgb, recompressed)

        extrema = difference.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        scale = 255.0 / max_diff
        ela_img = ImageEnhance.Brightness(difference).enhance(scale)

        ela_np = np.array(ela_img).astype(np.float32)
        if ela_np.ndim == 3:
            ela_np = np.mean(ela_np, axis=2)
        ela_np = ela_np / 255.0
        return ela_np


    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        file_name = self.df.iloc[index]["file_name"]
        label = self.df.iloc[index]["label"]
    
        rgb = Image.open(f"./{file_name}").convert("RGB")
        rgb_tensor = self.transform(rgb)

        combined_tensor_arr = [rgb_tensor]

        if self.return_prnu:        
            prnu = self.__extract_prnu__(rgb)
            prnu = cv2.resize(prnu, (self.img_size, self.img_size))
            prnu_tensor = torch.tensor(prnu).unsqueeze(0)
            combined_tensor_arr.append(prnu_tensor)

        if self.return_ela:
            ela = self.__extract_ela__(rgb)
            ela = cv2.resize(ela, (self.img_size, self.img_size))
            ela_tensor = torch.tensor(ela).unsqueeze(0)
            combined_tensor_arr.append(ela_tensor)

        return torch.cat(combined_tensor_arr, dim=0), torch.tensor(label, dtype=torch.int)

        