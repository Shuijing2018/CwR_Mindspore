from mindspore.nn.loss.loss import LossBase
import mindspore.ops as ops

class proposed_GCE_Loss(LossBase):
    def __init__(self, c, reduction="mean"):
        super(proposed_GCE_Loss, self).__init__(reduction)
        self.c = c

    def construct(self, output, target):
        label_rej = 0 * target + 10
        q = 0.7
        probit = ops.softmax(output, axis=1)+1e-18
        loss_vector_acc = (1 - ops.pow(ops.Gather()(probit, target, 0), q)) / q
        loss_vector_rej = (1 - ops.pow(ops.Gather()(probit, label_rej, 0), q)) / q
        loss_vector = (loss_vector_acc + (1 - self.c) * loss_vector_rej)
        loss = ops.mean(loss_vector)
        return self.get_loss(loss)
