from ImageDataset import ImageDataset 
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
import pandas as pd

IMAGES_NUM=10
BATCH_SIZE=32
IMG_SIZE=384
SEED=7

df = pd.read_csv("train.csv", nrows=IMAGES_NUM)

train_df, test_int_df = train_test_split(
    df,
    test_size=0.7,
    random_state=SEED,
    stratify=df["label"]
)

valid_df, test_df = train_test_split(
    test_int_df,
    test_size=0.5,
    random_state=SEED
)

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = ImageDataset(
    df=train_df,
    transform=train_transforms,
    return_ela=True,
    return_prnu=False,
    img_size=IMG_SIZE
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE)
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

for X, y in train_loader:
    print(X.shape)