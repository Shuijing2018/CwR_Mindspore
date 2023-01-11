# Classification with Rejection

This is the MindSpore implementation of CwR in the following paper.

Generalizing Consistent Multi-Class Classification with Rejection to be Compatible with Arbitrary Losses

# [CwR Description](#contents)

This paper derives a novel formulation for CwR that can be equipped with arbitrary loss functions while maintaining the theoretical guarantees. 

1) Show that K-class CwR is equivalent to a (K +1)-class classification problem on the original data distribution with an augmented class, and propose an empirical risk minimization formulation to solve this problem with an estimation error bound.

2)  Find a necessary and sufficient condition for the learning consistency of the surrogates constructed on our proposed formulation equipped with any classification-calibrated multi-class losses, where consistency means the surrogate risk minimization implies the target risk minimization for CwR.

# [Dataset](#contents)

Our experiments are conducted on three widely used benchmark datasets to test the performance of our CwR, which are Fashion-MNIST, SVHN and CIFAR-10.

# [Environment Requirements](#contents)

Framework

- [MindSpore](https://gitee.com/mindspore/mindspore)

For more information, please check the resources below：

- [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
- [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

```bash
python demo.py
```
