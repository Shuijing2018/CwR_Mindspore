import mindspore.common.dtype as mstype
import mindspore.dataset.engine as de

import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.dataset.vision as C
import mindspore.dataset.transforms as C2

def create_dataset(dataset_path, do_train, repeat_num=1, batch_size=32):

    ds = de.Cifar10Dataset(dataset_path, num_parallel_workers=4, shuffle=True)


    rescale = 1.0 / 255.0
    shift = 0.0

    # define map operations
    random_crop_op = C.RandomCrop((32, 32), (4, 4, 4, 4))
    random_horizontal_flip_op = C.RandomHorizontalFlip()

    rescale_op = C.Rescale(rescale, shift)
    change_swap_op = C.HWC2CHW()

    trans = []
    if do_train:
        trans += [random_crop_op, random_horizontal_flip_op]

    trans += [rescale_op, change_swap_op]

    type_cast_op = C2.TypeCast(mstype.int32)

    ds = ds.map(input_columns="label", operations=type_cast_op)
    ds = ds.map(input_columns="image", operations=trans)

    # apply shuffle operations
    ds = ds.shuffle(buffer_size=100)

    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)

    # apply dataset repeat operation
    ds = ds.repeat(repeat_num)

    return ds