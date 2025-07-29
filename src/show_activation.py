import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import numpy as np
import matplotlib.pyplot as plt
from nets import model_nvidia

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'dataset_dir', './data/datasets/driving_dataset',
    "Directory that stores input front-view images and steering angles.")
tf.app.flags.DEFINE_string(
    'model_file', './data/models/nvidia/model.ckpt',
    "Path to the trained model checkpoint.")

def _generate_feature_image(feature_map, grid_shape):
    # feature_map: H×W×C, grid_shape: [rows, cols] such that rows*cols = C
    h, w, c = feature_map.shape
    fmin = feature_map.min()
    fmax = feature_map.max()
    frange = fmax - fmin if (fmax - fmin) != 0 else 1.0

    out = np.zeros((h * grid_shape[0], w * grid_shape[1]), dtype=np.float32)
    idx = 0
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            patch = feature_map[:, :, idx]
            norm = (patch - fmin) / frange
            y0, y1 = i*h, (i+1)*h
            x0, x1 = j*w, (j+1)*w
            out[y0:y1, x0:x1] = norm
            idx += 1
    return out

def show_activation(_):
    img_path = f"{FLAGS.dataset_dir}/29649.jpg"
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = cv2.resize(rgb, (200, 66)) / 255.0  # (width, height)

    plt.figure('CNN Internal Activation')
    plt.subplot(2,1,1)
    plt.title('Input 3@66×200')
    plt.imshow(image)

    saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver.restore(sess, FLAGS.model_file)
        print("Model restored successfully.")

        conv1, conv2, conv3, conv4, conv5 = sess.run(
            [model_nvidia.h_conv1, model_nvidia.h_conv2,
             model_nvidia.h_conv3, model_nvidia.h_conv4,
             model_nvidia.h_conv5],
            feed_dict={model_nvidia.x: [image]}
        )

        f1_img = _generate_feature_image(conv1[0], [6, conv1.shape[3] // 6])
        f2_img = _generate_feature_image(conv2[0], [6, conv2.shape[3] // 6])

        def avg_act(act): 
            return act.mean(axis=3).squeeze(axis=0)
        a1, a2, a3, a4, a5 = map(avg_act, (conv1, conv2, conv3, conv4, conv5))

        def up(src, tgt):
            return cv2.resize(src, (tgt.shape[1], tgt.shape[0]), interpolation=cv2.INTER_AREA)

        m45 = a4 * up(a5, a4)
        m34 = a3 * up(m45, a3)
        m23 = a2 * up(m34, a2)
        m12 = a1 * up(m23, a1)

        sal = (m12 - m12.min()) / (m12.max() - m12.min() + 1e-8)
        sal_up = cv2.resize(sal, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)

        plt.subplot(2,2,3)
        plt.title('Layer1 Feature Maps')
        plt.imshow(f1_img, cmap='gray')

        plt.subplot(2,2,4)
        plt.title('Layer2 Feature Maps')
        plt.imshow(f2_img, cmap='gray')

    plt.show()

if __name__ == '__main__':
    tf.app.run(main=show_activation)
