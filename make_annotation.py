import os
import random

import numpy as np
from PIL import Image
from tqdm import tqdm

# -------------------------------------------------------#
#   想要增加测试集修改trainval_percent
#   修改train_percent用于改变验证集的比例 9:1
#
#   当前该库将测试集当作验证集使用，不单独划分测试集
# -------------------------------------------------------#
trainval_percent = 1
train_percent = 0.9
# -------------------------------------------------------#
#   指向数据集所在的文件夹
#   默认指向根目录下的radarsignal
# -------------------------------------------------------#
dataset_path = 'radarsignal'
# -------------------------------------------------------#
#   指向打印像素数量点，默认指向logs
# -------------------------------------------------------#
save_dir = 'logs'
# -------------------------------------------------------#
#   表示数据集的格式，PNG和NPY
#   默认情况下的格式为PNG
# -------------------------------------------------------#
dataset_form = 'PNG'

if __name__ == "__main__":
    if dataset_form == 'PNG':
        data_suffix = '.png'
    elif dataset_form == 'NPY':
        data_suffix = '.npy'
    else:
        raise AssertionError("Please specify the correct form: 'PNG' or 'NPY'.")
    random.seed(0)
    print("Generate txt in DataSets.")
    segfilepath = os.path.join(dataset_path, 'SegmentationClass')
    saveBasePath = os.path.join(dataset_path, 'DataSets')

    temp_seg = os.listdir(segfilepath)
    total_seg = []
    for seg in temp_seg:
        if seg.endswith(data_suffix):
            total_seg.append(seg)

    num = len(total_seg)
    list = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(list, tv)
    train = random.sample(trainval, tr)

    print("train and val size", tv)
    print("traub suze", tr)
    ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
    ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
    ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
    fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')

    for i in list:
        name = total_seg[i][:-4] + '\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()
    print("Generate txt in DataSets done.")

    print("Check datasets format, this may take a while.")
    print("检查数据集格式是否符合要求，这可能需要一段时间。")
    classes_nums = np.zeros([256], np.int)
    for i in tqdm(list):
        name = total_seg[i]
        png_file_name = os.path.join(segfilepath, name)
        if not os.path.exists(png_file_name):
            raise ValueError("未检测到标签图片%s，请查看具体路径下文件是否存在以及后缀是否为png。" % (png_file_name))

        png = np.array(Image.open(png_file_name), np.uint8)
        if len(np.shape(png)) > 2:
            print("标签图片%s的shape为%s，不属于灰度图或者八位彩图，请仔细检查数据集格式。" % (name, str(np.shape(png))))
            print("标签图片需要为灰度图或者八位彩图，标签的每个像素点的值就是这个像素点所属的种类。" % (
                name, str(np.shape(png))))

        classes_nums += np.bincount(np.reshape(png, [-1]), minlength=256)
    with open(os.path.join(save_dir, "pixel_type.txt"), 'a') as f:
        f.write("打印像素点的值与数量。")
        f.write("\n")
        f.write('-' * 37)
        f.write("\n")
        f.write("| %15s | %15s |" % ("Key", "Value"))
        f.write("\n")
        f.write('-' * 37)
        f.write("\n")
        for i in range(256):
            if classes_nums[i] > 0:
                f.write("| %15s | %15s |" % (str(i), str(classes_nums[i])))
                f.write("\n")
                f.write('-' * 37)
                f.write("\n")

    print("打印像素点的值与数量。")
    print('-' * 37)
    print("| %15s | %15s |" % ("Key", "Value"))
    print('-' * 37)
    for i in range(256):
        if classes_nums[i] > 0:
            print("| %15s | %15s |" % (str(i), str(classes_nums[i])))
            print('-' * 37)

    if classes_nums[255] > 0 and classes_nums[0] > 0 and np.sum(classes_nums[1:255]) == 0:
        print("检测到标签中像素点的值仅包含0与255，数据格式有误。")
        print("二分类问题需要将标签修改为背景的像素点值为0，目标的像素点值为1。")
    elif classes_nums[0] > 0 and np.sum(classes_nums[1:]) == 0:
        print("检测到标签中仅仅包含背景像素点，数据格式有误，请仔细检查数据集格式。")
