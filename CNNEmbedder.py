import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models

# ---------------------------
# CNN embedding (ResNet50 pool)
# ---------------------------
class CNNEmbedder:
    def __init__(self, model_name='resnet50'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
            # remove classifier/last layer, keep global pool output (2048)
            modules = list(model.children())[:-1]
            self.backbone = nn.Sequential(*modules).to(self.device)
        else:
            raise NotImplementedError("Only resnet50 currently")
        self.backbone.eval()
        self.transform = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    @torch.no_grad()
    def get(self, pil_img):
        x = self.transform(pil_img).unsqueeze(0).to(self.device)
        feat = self.backbone(x)  # [1, 2048, 1, 1]
        feat = feat.view(feat.size(0), -1)
        return feat.cpu().numpy().reshape(-1)  # 1D array