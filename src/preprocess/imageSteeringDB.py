import os
import cv2
import random
import numpy as np

class ImageSteeringDB(object):
    """Preprocess images of the road ahead and steering angles."""

    def __init__(self, data_dir):
        imgs = []
        angles = []

        # pointers for train & validation batches
        self.train_batch_pointer = 0
        self.val_batch_pointer = 0

        # read data.txt
        data_path = os.path.join(data_dir, "")
        with open(os.path.join(data_path, "data.txt"), 'r') as f:
            for line in f:
                parts = line.split()
                img_file = parts[0]
                angle_deg = float(parts[1])
                imgs.append(os.path.join(data_path, img_file))
                # convert steering angle to radians
                angles.append(angle_deg * np.pi / 180.0)

        # shuffle dataset
        combined = list(zip(imgs, angles))
        random.shuffle(combined)
        imgs, angles = zip(*combined)

        # split into train and validation sets (80/20)
        self.num_images = len(imgs)
        split_index = int(self.num_images * 0.8)
        self.train_imgs = imgs[:split_index]
        self.train_angles = angles[:split_index]
        self.val_imgs = imgs[split_index:]
        self.val_angles = angles[split_index:]

        self.num_train_images = len(self.train_imgs)
        self.num_val_images = len(self.val_imgs)

    def load_train_batch(self, batch_size):
        batch_imgs = []
        batch_angles = []
        for i in range(batch_size):
            idx = (self.train_batch_pointer + i) % self.num_train_images
            # read image (BGR by default)
            raw = cv2.imread(self.train_imgs[idx])
            # crop bottom 150 rows
            raw = raw[-150:, :, :]
            # convert BGR to RGB
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            # resize to 200x66 (width x height)
            img = cv2.resize(raw, (200, 66), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0
            batch_imgs.append(img)
            batch_angles.append([self.train_angles[idx]])

        self.train_batch_pointer += batch_size
        return batch_imgs, batch_angles

    def load_val_batch(self, batch_size):
        batch_imgs = []
        batch_angles = []
        for i in range(batch_size):
            idx = (self.val_batch_pointer + i) % self.num_val_images
            raw = cv2.imread(self.val_imgs[idx])
            raw = raw[-150:, :, :]
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            img = cv2.resize(raw, (200, 66), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0
            batch_imgs.append(img)
            batch_angles.append([self.val_angles[idx]])

        self.val_batch_pointer += batch_size
        return batch_imgs, batch_angles
