import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models

# ---------------------------
# CNN embedding (ResNet50 pool)
# ---------------------------
class CNNEmbedder:
    def __init__(self, model_name='resnet50'):
        # Select CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
            # Remove the final FC classifier → keep feature extractor only
            # Output of ResNet50 global avg pool is 2048-d
            modules = list(model.children())[:-1]
            self.backbone = nn.Sequential(*modules).to(self.device)
        else:
            raise NotImplementedError("Only resnet50 currently")
        
        # Put backbone in inference mode (BatchNorm, Dropout disabled)
        self.backbone.eval()

        # Preprocessing required by ImageNet-trained ResNet
        self.transform = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    @torch.no_grad()
    def get(self, pil_img):
        # Apply transforms and add batch dimension [1,3,224,224]
        x = self.transform(pil_img).unsqueeze(0).to(self.device)

        # Forward through backbone, output shape = [1, 2048, 1, 1]
        feat = self.backbone(x)  # [1, 2048, 1, 1]

         # Flatten to [1, 2048]
        feat = feat.view(feat.size(0), -1)

        # Return a 1D numpy vector of length 2048
        return feat.cpu().numpy().reshape(-1)