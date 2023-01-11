from models import resnet34
from mindspore.nn.optim import Adam
from mindspore import set_context, PYNATIVE_MODE
# set_context(mode=PYNATIVE_MODE)
from mindspore.train.model import Model

from mindspore.train.callback import LossMonitor, TimeMonitor
import utils_data
import utils_algo
import warnings
import argparse
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(
    prog='Demo file for consistent CwR',
    usage='Demo file',
    description='A simple demo file with CIFAR-10 dataset.',
    epilog='end',
    add_help=True)

parser.add_argument('-lr', '--learning_rate', help='optimizer\'s learning rate', default=1e-3, type=float)
parser.add_argument('-bs', '--batch_size', help='batch_size', default=1024, type=int)
parser.add_argument('-e', '--epochs', help='number of epochs', type=int, default=200)
parser.add_argument('-wd', '--weight_decay', help='weight decay', default=1e-4, type=float)
parser.add_argument('-c', '--cost', help='cost', default=1e-1, type=float)

args = parser.parse_args()

model = []
dataset = utils_data.create_dataset(dataset_path='cifar10/train', do_train=True,
                                 repeat_num=args.epochs, batch_size=args.batch_size)
step_size = dataset.get_dataset_size()
cost = args.cost

net = resnet34(class_num=11)
loss = utils_algo.proposed_GCE_Loss(args.cost)
for T in range(1):
    opt = Adam(net.trainable_params(), learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'},loss_scale_manager=None, amp_level="O2",keep_batchnorm_fp32=False)

    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    model.train(args.epochs, dataset, callbacks=cb)

