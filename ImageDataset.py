import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import pywt
import io
from PIL import Image, ImageChops, ImageEnhance

class ImageDataset(Dataset):

    def __init__(self, csv_path, transform=None, return_prnu=False, 
                return_ela=False, ela_quality=90, img_num=10):
        """
        csv_path: Path for CSV with data from the images ["file_name", "label"]
        transform: torchvision transforms
        return_prnu: True -> return (RGB+PRNU) 4-channel tensor
        return_ela: True -> return ELA image
        ela_quality: JPEG quality for recompression
        """
        self.df = pd.read_csv(csv_path, nrows=img_num)
        print(len(self.df))
        self.transform = transform
        self.return_prnu = return_prnu
        self.return_ela = return_ela
        self.ela_quality = ela_quality
    
    def __extract_prnu__(self,rgb):
        """
        rgb: rgb image
        Uses safe PRNU extraction (wavelet denoise -> noise residual).
        """
        gray = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2GRAY).astype(np.float32)

        # Wavelet denoise (remove content, leave noise)
        coeffs = pywt.wavedec2(gray, 'db4', level=4)
        coeffsH = list(coeffs)
        coeffsH[0] *= 0
        denoised = pywt.waverec2(coeffsH, 'db4')

        Image.fromarray(denoised).show()

        # Noise residual
        noise = gray - denoised        

        # Zero - mean
        noise = noise - np.mean(noise)

        # Normalize by intensity (avoid division by zero)
        prnu = noise / (gray + 1e-6)
        return prnu.astype(np.float32)
    
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

        ela_img.show()
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
        rgb.show()
        print(np.array(rgb).astype(np.float32).shape)
        rgb_tensor = torch.tensor(np.array(rgb).astype(np.float32) / 255.0 ) 
        
        #TODO transforms

        # Add features and create tensor
        combined_tensor_arr = [rgb_tensor]

        if self.return_prnu:        
            prnu = self.__extract_prnu__(rgb)
            prnu_tensor = torch.tensor(prnu).unsqueeze(0)
            combined_tensor_arr.append(prnu_tensor)

        if self.return_ela:
            ela = self.__extract_ela__(rgb)
            ela_tensor = torch.tensor(ela).unsqueeze(0)
            combined_tensor_arr.append(ela_tensor)

        print(combined_tensor_arr)
        return torch.cat(combined_tensor_arr, dim=0), torch.tensor(label, dtype=torch.int)

        