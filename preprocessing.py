import random
from pathlib import Path
import numpy as np
from tensorflow import keras


class DataGen(object):

    def __init__(self,
                 rotation_range=30,
                 width_shift_range=0.3,
                 height_shift_range=0.3,
                 brightness_range=(0.5, 1.5),
                 shear_range=0.1,
                 zoom_range=(0.8, 1.2),
                 fill_mode='nearest',
                 horizontal_flip=True,
                 vertical_flip=True,
                 target_size=(105, 105)
    ):
        self._target_size = target_size
        self._image_gen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            brightness_range=brightness_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            fill_mode=fill_mode,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
        )

        self._loaded_images = {}

    def flow(self, images_dir, batch_size=32, seed=None, shuffle=True):
        if seed is not None:
            random.seed(seed)

        dataset = [(i, pp.name.split('.')[0], str(pp))
                   for i, pp in enumerate(Path(images_dir).glob('*.png'))]
        dataset_iterator = self._iter_training_dataset(dataset, shuffle)

        same = True
        while True:
            pairs = np.zeros((batch_size, 2, *self._target_size, 3))
            labels = np.zeros((batch_size, 1))

            for i in range(batch_size):
                index, name, img = next(dataset_iterator)

                trans_img_left = self._image_gen.random_transform(img)
                if same:
                    trans_img_right = self._image_gen.random_transform(img)
                    label = 1
                else:
                    other_index = (index + random.randint(
                        1, len(dataset) - 1)) % len(dataset)
                    img_right = self._load_img(dataset[other_index][2])
                    trans_img_right = self._image_gen.random_transform(img_right)
                    label = 0

                pairs[i][0] = trans_img_left
                pairs[i][1] = trans_img_right
                labels[i] = label

                same = not same

            yield pairs, labels

    def _iter_training_dataset(self, dataset, shuffle):
        training_dataset_iter = self._iter_dataset(dataset, shuffle)
        for i, name, img in training_dataset_iter:
            yield i, name, img
            yield i, name, img

    def _iter_dataset(self, dataset, shuffle):
        # shuffle dataset
        loaded_images = {}

        while True:
            for i, name, img_path in dataset:
                yield i, name, self._load_img(img_path)

    def _load_img(self, path):
        img = self._loaded_images.get(path)
        if img is None:
            img = keras.preprocessing.image.load_img(
                path,
                target_size=self._target_size,
                color_mode='rgb'
            )
            img = keras.preprocessing.image.img_to_array(img)
            self._loaded_images[path] = img

        return img
