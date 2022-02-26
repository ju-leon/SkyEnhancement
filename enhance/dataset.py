import tensorflow as tf
import albumentations as albu
import os
import glob
import cv2
import numpy as np
import tqdm

IMAGE_FORMAT = ".jpg"


def reduce_haze(image, **kwargs):
    return (np.power(image / 255, 4) * 255).astype(np.uint8)


class Dataset:
    def get_filename(self, string):
        return os.path.splitext(os.path.basename(string))[0]

    def __init__(self, images_dir, image_size=256) -> None:
        self.image_size = image_size

        self.images_dir = images_dir
        self.ids = list(map(self.get_filename, glob.glob(
            os.path.join(images_dir, '*' + IMAGE_FORMAT))))

        self.images_fps = [os.path.join(
            images_dir, image_id + IMAGE_FORMAT) for image_id in self.ids]

        self.augmentation = self.get_training_augmentation()

    def __getitem__(self, i):
        return self.get_sample(self.images_fps[i])

    def get_sample(self, path):
        image = cv2.imread(path)

        sample = self.augmentation(image=image, mask=image)
        image_input, image_output = sample['image'], sample['mask']

        # Normalize images to range [-1, 1]
        image_input = (image_input / 127.5) - 1
        image_output = (image_output / 127.5) - 1

        return image_input.astype(np.float32), image_output.astype(np.float32)

    def get_dataset(self, patches_per_image=4):
        images_input = []
        images_output = []
        for image_path in self.images_fps:
            for _ in tqdm(range(patches_per_image)):
                image_input, image_output = self.get_sample(path=image_path)
                images_input.append(image_input)
                images_output.append(image_output)

        return images_input, images_output

    def get_training_augmentation(self):
        train_transform = [
            albu.HorizontalFlip(p=0.5),

            albu.augmentations.geometric.transforms.ShiftScaleRotate(
                scale_limit=[-0.5, 3], rotate_limit=45, shift_limit=0.1, p=1, border_mode=0),

            albu.augmentations.transforms.PadIfNeeded(min_height=self.image_size,
                                                      min_width=self.image_size,
                                                      always_apply=True,
                                                      border_mode=0),

            albu.augmentations.crops.transforms.RandomCrop(height=self.image_size,
                                                           width=self.image_size,
                                                           always_apply=True),

            albu.augmentations.transforms.GaussNoise(p=0.4),
            albu.augmentations.transforms.ISONoise(p=0.3),
            albu.augmentations.geometric.transforms.Perspective(p=0.2),
            albu.augmentations.transforms.RandomFog(p=0.05),

            albu.OneOf(
                [
                    albu.augmentations.transforms.RandomToneCurve(p=1),
                    albu.augmentations.transforms.RGBShift(p=1),
                    albu.augmentations.transforms.Lambda(
                        image=reduce_haze, p=1)

                ],
                p=0.5
            ),

            albu.OneOf(
                [
                    albu.augmentations.transforms.CLAHE(p=1),
                    albu.augmentations.transforms.RandomGamma(p=1),
                ],
                p=0.9,
            ),

            albu.OneOf(
                [
                    albu.augmentations.transforms.Sharpen(p=1),
                    albu.augmentations.transforms.Blur(blur_limit=3, p=1),
                    albu.augmentations.transforms.MotionBlur(
                        blur_limit=3, p=1),
                ],
                p=0.9,
            ),

            albu.OneOf(
                [
                    albu.augmentations.transforms.RandomBrightnessContrast(
                        brightness_limit=[-0.6, 0.1], p=1),
                    albu.augmentations.transforms.HueSaturationValue(p=1),
                ],
                p=0.9,
            ),
        ]
        return albu.Compose(train_transform)
