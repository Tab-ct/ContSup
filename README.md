

# ContSup-Pytorch

Greedy Local Learning with Context Supply for training deep networks.


## Introduction
We proposed the Context Supply (ContSup) scheme after concluding that existing Greedy Local Learning (GLL) schemes are incapable of effectively addressing the confirmed habit dilemma. ContSup allows local modules to retain more information via additional context, thereby improving the theoretical effectiveness of final performance. ContSup can significantly lower the memory footprint of GPUs while retaining the same level of performance; steady performance can be maintained even as the number of isolated modules increases.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Get Started
To train ResNet-32/ResNet-110 with ContSup, run this command:
```
python train.py --dataset <name_dataset: cifar10/svhn/stl10> --model resnet --layers <32/110> --local_module_num <num_partition: support for 2/4/8/16> --local_loss_mode <cross_entropy/contrast> --context_mode <context_mode: e.g. R0/R1/E/R1E>
```

For example, to train a 16-partitioned ResNet-32 on CIFAR-10 with ContSup\[E\](softmax), run:
```
python train.py --dataset cifar10 --model resnet --layers 32 --local_module_num 16 --local_loss_mode cross_entropy --context_mode E
```

For more examples, please check out the "command.sh" file.

## Results Reproduction

With the provided codes, you may reproduce all of the following experiments from the paper:

- Experiment #1 (Table 1): Performance of baseline\[R0\], Contsup\[E\], Contsup\[R1\] and Contsup\[R1E\] on CIFAR-10 with 16-partitioned ResNet-32.
- Experiment #2 (Table 2): Performance of Contsup\[E\](contrast/softmax) and Contsup\[R1E\](contrast/softmax) on CIFAR-10 with $K$-partitioned ResNet-32.
- Experiment #3 (Table 3): Performance of Contsup\[R1E\](contrast/softmax) on CIFAR-10/SVHN/STL-10 with $K$-partitioned ResNet-110.
- Experiment #4 (Figure 4): Error rates (\%)  and gpu memory-cost (GB) of ContSup\[E\]* (memory balance) with ResNet-32/110 on CIFAR-10.
- Experiment #5 (Figure 5): Ablation studies of ContSup with $K$-partitioned ResNet-32 on CIFAR-10.
- Experiment #6 (Table 4): Error rates (\%) and gpu memory-cost (GB) of different ContSup modes with 8/16-partitioned ResNet-32 on CIFAR-10.

Please refer to "command.sh" for all the commands needed to reproduce the experiments that yielded the reported results.

## Reference
[1] C. Yu, F. Zhang, H. Ma, A. Wang, and E. Li, “Go beyond End-to-End Training: Boosting Greedy Local Learning with Context Supply.” arXiv, Dec. 12, 2023. doi: 10.48550/arXiv.2312.07636. Available: https://arxiv.org/abs/2312.07636.

