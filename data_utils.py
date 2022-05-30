import sys
from glob import glob
import concurrent.futures
import multiprocessing
import os
import numpy as np
import cv2
import time
import random
import argparse
from PIL import Image, ImageEnhance, ImageOps


def create_directory(path_list):
    for path in path_list:
        print("Creating Direcory: {}".format(path))
        if os.path.exists(path):
            print("[ERROR] Path Already Exists : {}".format(path))
            # sys.exit()
        else:
            os.mkdir(path)


def generate_image_sets(src_path, target_path):
    src_img_list = [y for x in os.walk(src_path) for y in glob(os.path.join(x[0], '*.jpg'))]
    src_lbl_list = [x.replace('.jpg', '.png') for x in src_img_list]
    src_lbl_list = [x.replace('clip_img', 'matting') for x in src_lbl_list]
    src_lbl_list = [x.replace('clip', 'matting') for x in src_lbl_list]

    target_img_path = os.path.join(target_path, "images")
    target_lable_path = os.path.join(target_path, "labels")
    create_directory([target_path, target_lable_path, target_img_path])

    with open('{}/train.txt'.format(target_path), 'w') as train_file:
        with open('{}/val.txt'.format(target_path), 'w') as val_file:
            for src_img_name in src_img_list:
                base_name = os.path.split(src_img_name)[1]
                entry_name = base_name.replace('.jpg', "") + '\n'
                src_lbl_name = src_img_name.replace('.jpg', '.png')
                src_lbl_name = src_lbl_name.replace('clip_img', 'matting')
                src_lbl_name = src_lbl_name.replace('clip', 'matting')

                tgt_img_name = os.path.join(target_img_path, base_name)
                tgt_lbl_name = os.path.join(target_lable_path, base_name.replace('.jpg', '.png'))

                src_img = cv2.imread(src_img_name)
                cv2.imwrite(tgt_img_name, src_img)
                print(src_lbl_name)
                src_lbl = cv2.imread(src_lbl_name, cv2.IMREAD_UNCHANGED)

                lbl_alpha = src_lbl[:, :, 3]
                tgt_lbl = np.zeros((src_lbl.shape[0], src_lbl.shape[1], 3), dtype=np.float32)
                tgt_lbl[:, :, 0] = lbl_alpha
                tgt_lbl[:, :, 1] = lbl_alpha
                tgt_lbl[:, :, 2] = lbl_alpha
                tgt_lbl = cv2.cvtColor(tgt_lbl, cv2.COLOR_BGR2GRAY)
                ret, thresh_lbl = cv2.threshold(tgt_lbl, 127, 1, cv2.THRESH_BINARY)
                cv2.imwrite(tgt_lbl_name, thresh_lbl)
                if random.randint(0, 6) == 0:
                    val_file.write(entry_name)
                else:
                    train_file.write(entry_name)


# random brightness control
def rnd_bright(image, rnd_min=-1.0, rnd_max=1.0):
    rnd_num = random.uniform(rnd_min, rnd_max)
    factor = 2 ** rnd_num
    enhancer = ImageEnhance.Brightness(image)
    br_image = enhancer.enhance(factor)
    return br_image


# random contrast control
def rnd_contrast(image, rnd_min=-1.0, rnd_max=1.0):
    rnd_num = random.uniform(rnd_min, rnd_max)
    factor = 2 ** rnd_num
    enhancer = ImageEnhance.Contrast(image)
    ct_image = enhancer.enhance(factor)
    return ct_image


# equalize with probability
def equalize_prob(image, prob=1.0):
    # probability
    choice = np.random.choice(2, None, p=[prob, (1.0 - prob)])
    if choice == 1:
        return image

    # equalize
    eq_image = ImageOps.equalize(image)

    return eq_image


# solarize with probability and magnitude
def solarize_prob(image, prob=1.0, mag=0):
    # probability
    choice = np.random.choice(2, None, p=[prob, (1.0 - prob)])
    if choice == 1:
        return image

    # magnitude
    rnd_mag = random.randint(0, mag)
    if rnd_mag >= 0 and rnd_mag < 10:
        threshold = 255 * (10 - rnd_mag) / 10
    else:
        threshold = 255

    # solarize
    sol_image = ImageOps.solarize(image, threshold)

    return sol_image


# posterize with probability and magnitude
def posterize_prob(image, prob=1.0, mag=0):
    # probability
    choice = np.random.choice(2, None, p=[prob, (1.0 - prob)])
    if choice == 1:
        return image

    # magnitude
    rnd_mag = random.randint(0, mag)
    if rnd_mag >= 0 and rnd_mag < 10:
        n_bits = 4 + int(4 * (10 - rnd_mag) / 10)
    else:
        n_bits = 8

    # posterize
    pos_image = ImageOps.posterize(image, n_bits)

    return pos_image


# rotate image with random
def rotate_rand(image, label, deg_min=-5.0, deg_max=5.0):
    # random angle
    angle = random.triangular(deg_min, deg_max)

    # rotate image
    rot_image = image.rotate(-angle, resample=Image.BICUBIC, expand=1, fillcolor='gray')
    rot_label = label.rotate(-angle, resample=Image.BICUBIC, expand=1, fillcolor='black')

    return rot_image, rot_label


# flip image
def flip_img(image, label):
    # flip image
    flip_image = cv2.flip(image, 1)
    flip_label = cv2.flip(label, 1)

    return flip_image, flip_label


# augmentation process
def aug_img(img_fn):
    global BG_LEN, BG_LIST, BG_PATH, IMG_DIM, SRC_PATH, TGT_PATH
    try:
        # read images and labels
        src_img_path = os.path.join(SRC_PATH, 'images')
        src_lbl_path = os.path.join(SRC_PATH, 'labels')
        src_img_fn = os.path.join(src_img_path, img_fn)
        base = img_fn.split('.')[0]
        src_lbl_fn = os.path.join(src_lbl_path, base + '.png')
        src_img = cv2.imread(src_img_fn)
        src_lbl = cv2.imread(src_lbl_fn, cv2.IMREAD_UNCHANGED)

        src_h, src_w, src_c = src_img.shape

        # loop
        loop_cnt = 5

        for lidx in range(loop_cnt):
            # convert to RGB (for PIL)
            img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
            lbl = src_lbl
            img_pil = Image.fromarray(img)
            lbl_pil = Image.fromarray(lbl)

            # =============================================
            # augmentation policies
            # =============================================

            # sub-policy distribution
            #   - reference: AutoAugment
            #   - best 5 sub-policies on ImageNet:
            #       1. (Equalize:  0.4, 4, Rotate:    0.8, 8)
            #       2. (Solarize:  0.6, 3, Equalize:  0.6, 7)
            #       3. (Posterize: 0.8, 5, Equalize:  1.0, 2)
            #       4. (Rotate:    0.2, 3, Solarize:  0.6, 8)
            #       5. (Equalize:  0.6, 8, Posterize: 0.4, 6)
            #   - distribution is temporary
            sub_policy_idx = np.random.choice(6, None, p=[0.2, 0.2, 0.2, 0.1, 0.2, 0.1])
            # print('sub-policy index: ', sub_policy_idx)

            lbl_tr = lbl_pil

            if sub_policy_idx == 0:
                img_tr1, lbl_tr = rotate_rand(img_pil, lbl_pil, -15.0, 15.0)
                img_tr2 = rnd_bright(img_tr1, -0.6, 0.5)
            elif sub_policy_idx == 1:
                img_tr1 = rnd_contrast(img_pil, -0.4, 0.4)
                img_tr2 = rnd_bright(img_tr1, -0.6, 0.4)
            elif sub_policy_idx == 2:
                img_tr1, lbl_tr = rotate_rand(img_pil, lbl_pil, -15.0, 15.0)
                img_tr2 = rnd_contrast(img_tr1, -0.4, 0.4)
            elif sub_policy_idx == 3:
                img_tr1 = rnd_bright(img_pil, -0.5, 0.4)
                img_tr2 = solarize_prob(img_tr1, 0.5, 1)
            elif sub_policy_idx == 4:
                img_tr1 = rnd_bright(img_pil, -0.6, 0.5)
                img_tr2 = posterize_prob(img_tr1, 0.8, 3)
            else:
                img_tr1 = equalize_prob(img_pil, 0.4)
                img_tr2 = img_tr1

            # return to BGR (for OpenCV)
            img_np = np.asarray(img_tr2)
            img_aug = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            lbl_np = np.asarray(lbl_tr)
            ret, lbl_aug = cv2.threshold(lbl_np, 0.5, 1, cv2.THRESH_BINARY)

            if np.random.randint(0, 2) == 1:
                img_aug, lbl_aug = flip_img(img_aug, lbl_aug)

            h_aug, w_aug, c_aug = img_aug.shape

            # additive noise
            mean = 0
            var = np.clip(np.random.normal(100.0, 100.0), 0.10, 1000.0)
            sigma = var ** 0.5
            gaussian = np.random.normal(mean, sigma, (h_aug, w_aug, c_aug))
            img_float = img_aug.astype(np.float32)
            img_noise_float = np.clip(img_float + gaussian, 0.0, 255.0)
            img_noise = img_noise_float.astype(np.uint8)
            noise_h, noise_w, _ = img_noise.shape

            # 4x4 extended canvas, centered image, and cut off
            h_ext = src_h * 4
            w_ext = src_h * 4

            # background images
            bg_pick = False

            while bg_pick == False:
                bg_idx = random.randint(0, BG_LEN - 1)
                bg_fn = BG_LIST[bg_idx]
                bg_img_fn = os.path.join(BG_PATH, bg_fn)
                bg_img = cv2.imread(bg_img_fn)
                if bg_img is None:
                    continue

                bg_h, bg_w, bg_c = bg_img.shape

                if bg_h > h_ext and bg_w > w_ext:
                    bg_pick = True

            h_off_max = bg_h - h_ext - 1
            w_off_max = bg_w - w_ext - 1

            h_off = random.randint(0, h_off_max)
            w_off = random.randint(0, w_off_max)

            img_ext = bg_img[h_off:h_off + h_ext, w_off:w_off + w_ext, :]
            img_ext[int(1.5 * src_h):int(1.5 * src_h) + src_h,
            int(2.0 * src_h - 0.5 * src_w):int(2.0 * src_h + 0.5 * src_w), :] = img_noise[
                                                                                int(noise_h / 2) - int(src_h / 2):int(
                                                                                    noise_h / 2) + int(src_h / 2),
                                                                                int(noise_w / 2) - int(src_w / 2):int(
                                                                                    noise_w / 2) + int(src_w / 2),
                                                                                :]

            lbl_ext = np.zeros((h_ext, w_ext), dtype=np.uint8)
            lbl_ext[int(1.5 * src_h):int(1.5 * src_h) + src_h,
            int(2.0 * src_h - 0.5 * src_w):int(2.0 * src_h + 0.5 * src_w)] = lbl_aug[
                                                                             int(noise_h / 2) - int(src_h / 2):int(
                                                                                 noise_h / 2) + int(src_h / 2),
                                                                             int(noise_w / 2) - int(src_w / 2):int(
                                                                                 noise_w / 2) + int(src_w / 2)]

            ratio_crop = np.random.uniform(low=1.0, high=2.0)

            h_crop = int(src_h * ratio_crop)
            w_crop = int(src_w * ratio_crop * 4.0 / 3.0)

            y_min = max(0, 2 * src_h + int(0.5 * src_h) - h_crop - 1)
            y_max = min(2 * src_h - int(0.5 * src_h), 4 * src_h - h_crop)
            x_min = max(0, 2 * src_h + int(0.5 * src_w) - w_crop - 1)
            x_max = min(2 * src_h - int(0.5 * src_w), 4 * src_h - w_crop)

            x_off = int(np.random.triangular(x_min, 2 * src_h - (w_crop / 2), x_max))
            y_off = np.random.randint(y_min, y_max)

            img_crop = np.zeros((h_crop, w_crop)).astype(np.float32)
            img_crop = img_ext[y_off:y_off + h_crop, x_off:x_off + w_crop, :]
            lbl_crop = np.zeros((h_crop, w_crop)).astype(np.float32)
            lbl_crop = lbl_ext[y_off:y_off + h_crop, x_off:x_off + w_crop]

            # resize to 160x160
            img_new = cv2.resize(img_crop, (int(IMG_DIM), int(IMG_DIM)), interpolation=cv2.INTER_AREA)
            lbl_new = cv2.resize(lbl_crop, (int(IMG_DIM), int(IMG_DIM)), interpolation=cv2.INTER_AREA)

            tgt_img_path = os.path.join(TGT_PATH, 'images')
            tgt_lbl_path = os.path.join(TGT_PATH, 'labels')

            # target file name
            tgt_img_fn = os.path.join(tgt_img_path, base + '_' + str(lidx) + '.jpg')
            tgt_lbl_fn = os.path.join(tgt_lbl_path, base + '_' + str(lidx) + '.png')

            cv2.imwrite(tgt_img_fn, img_new)
            cv2.imwrite(tgt_lbl_fn, lbl_new)
    except Exception as e:
        print("There was error: {}".format(e))


def run_augmentaion(src_path, target_path, background_path, dim=160):
    global BG_LEN, BG_LIST, BG_PATH, IMG_DIM, SRC_PATH, TGT_PATH

    target_img_path = os.path.join(target_path, "images")
    target_lable_path = os.path.join(target_path, "labels")
    create_directory([target_path, target_lable_path, target_img_path])

    IMG_DIM = dim
    BG_PATH = background_path
    SRC_PATH = src_path
    TGT_PATH = target_path
    BG_LIST = os.listdir(BG_PATH)
    BG_LEN = len(BG_LIST)

    start = time.time()

    src_img_path = os.path.join(src_path, 'images')
    img_list = os.listdir(src_img_path)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(aug_img, img_list)
    '''
    img_list = os.listdir(src_img_path)
    for img in img_list:
        aug_img(img)
    '''
    finish = time.time()
    duration = finish - start

    time.strftime("%H:%M:%S", time.gmtime(duration))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, help="Generate/Augment")
    main_args, argv = parser.parse_known_args(sys.argv[1:3])

    if main_args.mode == "Generate":
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_data", type=str, required=True, default="", help="Input data root path")
        parser.add_argument("--output_data", type=str, required=True, default="", help="Output data root path")
        args, argv = parser.parse_known_args()
        generate_image_sets(args.input_data, args.output_data)

    if main_args.mode == "Augment":
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_data", type=str, required=True, default="", help="Input data root path")
        parser.add_argument("--output_data", type=str, required=True, default="", help="Output data root path")
        parser.add_argument("--background_data", type=str, required=True, default="", help="Background data path")
        parser.add_argument("--output_dim", type=str, required=False, default="160", help="Output dim")
        args, argv = parser.parse_known_args()
        run_augmentaion(args.input_data, args.output_data, args.background_data, int(args.output_dim))
