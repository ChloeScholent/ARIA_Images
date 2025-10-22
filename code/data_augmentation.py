import os
from torchvision.utils import save_image
from PIL import Image
from torchvision import transforms
import cv2 as cv


data_augmentation = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=1),
    transforms.RandomRotation(degrees=45),
    transforms.ToTensor()
])

folder_to_transform = "powerlifting/train/transformed_images/"
output_folder = "powerlifting/train/images/"

for image in os.listdir(folder_to_transform):
    image_path = os.path.join(folder_to_transform, image)
    name, ext = os.path.splitext(image)
    img = cv.imread(image_path)[:,:,::-1]
    augmented_img = data_augmentation(img)
    out_file = output_folder + name + "trans" + ext
    save_image(augmented_img, out_file)


