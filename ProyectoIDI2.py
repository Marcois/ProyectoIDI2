from ImageDataset import ImageDataset 
from torch.utils.data import DataLoader, random_split

IMAGES_NUM=10
BATCH_SIZE=32
dataset = ImageDataset(
    csv_path="train.csv",
    return_ela=True,
    return_prnu=True,
    img_num=IMAGES_NUM
)

train_size = int(0.5 * IMAGES_NUM)
validation_size = int(0.2 * IMAGES_NUM)
test_size = int(IMAGES_NUM - train_size - validation_size)

print(train_size, validation_size, test_size, dataset.__len__())
train_dataset, validation_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

for X, y in train_loader:
    print(X)