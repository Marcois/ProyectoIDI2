from ImageDataset import ImageDataset 
from torch.utils.data import DataLoader, random_split

IMAGES_NUM=10

dataset = ImageDataset(
    csv_path="train.csv",
    return_ela=True
)

train_size = int(0.7 * IMAGES_NUM)
validation_size = int(0.15 * IMAGES_NUM)
test_size = IMAGES_NUM - train_size - validation_size

train_dataset, validation_dataset, test_dataset = random_split(dataset,[train_size, test_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(train_dataset, batch_size=32)
test_loader = DataLoader(train_dataset, batch_size=32)

for X, y in train_loader:
    print(X)