import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class ExerciseDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(image_dir))

        # Map class names to numeric labels
        self.class_map = {"bench": 0, "squat": 1, "deadlift": 2}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Extract label from filename
        if "bench" in img_name.lower():
            label = self.class_map["bench"]
        elif "squat" in img_name.lower():
            label = self.class_map["squat"]
        elif "deadlift" in img_name.lower():
            label = self.class_map["deadlift"]
        else:
            raise ValueError(f"Unknown class in filename: {img_name}")

        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

<<<<<<< HEAD:code/dataset.py
powerlifting_dataset = ExerciseDataset("dataset/", transform=transform)
=======
powerlifting_dataset = ExerciseDataset("powerlifting/train/images", transform=transform)
>>>>>>> 899693a (update):code/correct_dataset.py
