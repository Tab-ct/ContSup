########################################################################
###### Table 1: Results with 16-partitioned ResNet-32 on CIFAR-10 ######
########################################################################

########################
####   Quick Start  ####
########################

CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 16 --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R0
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 16 --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode E
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 16 --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 16 --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E



#########################################################
###### Table 2: Results with ResNet-32 on CIFAR-10 ######
#########################################################

## ResNet-32 + ContSup[E](contrast) => CIFAR-10 ##

CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 2  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode E
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 4  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode E
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 8  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode E
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 16 --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode E

## ResNet-32 + ContSup[E](softmax) => CIFAR-10 ##

CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 2  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode E
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 4  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode E
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 8  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode E
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 16 --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode E

## ResNet-32 + ContSup[R1E](contrast) => CIFAR-10 ##

CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 2  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 4  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 8  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 16 --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E

## ResNet-32 + ContSup[R1E](softmax) => CIFAR-10 ##

CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 2  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 4  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 8  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 16 --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E



######################################################################
###### Table 3: Results with ResNet-110 on CIFAR-10/STL-10/SVHN ######
######################################################################

## ResNet-110 + ContSup[R1E](contrast) => CIFAR-10 ##

CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 2  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E --h_reconstruct True
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 4  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E --h_reconstruct True
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 8  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E --h_reconstruct True
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 16 --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E --h_reconstruct True

## ResNet-110 + ContSup[R1E](softmax) => CIFAR-10 ##

CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 2  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E --h_reconstruct True
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 4  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E --h_reconstruct True
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 8  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E --h_reconstruct True
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 16 --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E --h_reconstruct True

## ResNet-110 + ContSup[R1E](contrast) => STL-10 ##

CUDA_VISIBLE_DEVICES=0 python train.py --dataset stl10 --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 2  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E --h_reconstruct True
CUDA_VISIBLE_DEVICES=0 python train.py --dataset stl10 --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 4  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E --h_reconstruct True
CUDA_VISIBLE_DEVICES=0 python train.py --dataset stl10 --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 8  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E --h_reconstruct True
CUDA_VISIBLE_DEVICES=0 python train.py --dataset stl10 --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 16 --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E --h_reconstruct True

## ResNet-110 + ContSup[R1E](softmax) => STL-10 ##

CUDA_VISIBLE_DEVICES=0 python train.py --dataset stl10 --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 2  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E --h_reconstruct True
CUDA_VISIBLE_DEVICES=0 python train.py --dataset stl10 --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 4  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E --h_reconstruct True
CUDA_VISIBLE_DEVICES=0 python train.py --dataset stl10 --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 8  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E --h_reconstruct True
CUDA_VISIBLE_DEVICES=0 python train.py --dataset stl10 --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 16 --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E --h_reconstruct True

## ResNet-110 + ContSup[R1E](contrast) => SVHN ##

CUDA_VISIBLE_DEVICES=0 python train.py --dataset svhn --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 2  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E --h_reconstruct True
CUDA_VISIBLE_DEVICES=0 python train.py --dataset svhn --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 4  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E --h_reconstruct True
CUDA_VISIBLE_DEVICES=0 python train.py --dataset svhn --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 8  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E --h_reconstruct True
CUDA_VISIBLE_DEVICES=0 python train.py --dataset svhn --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 16 --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E --h_reconstruct True

## ResNet-110 + ContSup[R1E](softmax) => SVHN ##

CUDA_VISIBLE_DEVICES=0 python train.py --dataset svhn --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 2  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E --h_reconstruct True
CUDA_VISIBLE_DEVICES=0 python train.py --dataset svhn --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 4  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E --h_reconstruct True
CUDA_VISIBLE_DEVICES=0 python train.py --dataset svhn --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 8  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E --h_reconstruct True
CUDA_VISIBLE_DEVICES=0 python train.py --dataset svhn --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 16 --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E --h_reconstruct True



##############################################################
###### Figure 4: Results with ResNet-32/110 on CIFAR-10 ######
##############################################################

## ResNet-32 + ContSup[E]* (memory balance) => CIFAR-10 ##

CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 2  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode E
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 4  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode E
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 8  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode E
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 16 --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode E


## ResNet-110 + ContSup[E]* (memory balance) => CIFAR-10 ##

CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 2  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f --balanced_memory True  --context_mode E
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 3  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f --balanced_memory True  --context_mode E
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 110 --droprate 0.0 --cos_lr --local_module_num 4  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f --balanced_memory True  --context_mode E



#################################################################
###### Figure 5: Ablation study with ResNet-32 on CIFAR-10 ######
#################################################################

## ResNet-32 + GDL => CIFAR-10 ##
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 2  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R0
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 4  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R0
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 8  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R0
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 16 --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R0
## ResNet-32 + ContSup[E] => CIFAR-10 #
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 2  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode E
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 4  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode E
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 8  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode E
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 16 --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode E
## ResNet-32 + ContSup[R1] => CIFAR-10 ##
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 2  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 4  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 8  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 16 --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1
## ResNet-32 + ContSup[R1E] => CIFAR-10 ##
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 2  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 4  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 8  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 16 --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E

## ResNet-32 + GDL(decoder) => CIFAR-10 ##
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 2  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R0 --h_reconstruct True
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 4  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R0 --h_reconstruct True
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 8  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R0 --h_reconstruct True
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 16 --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R0 --h_reconstruct True
## ResNet-32 + ContSup[E](decoder) => CIFAR-10 ##
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 2  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode E --h_reconstruct True
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 4  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode E --h_reconstruct True
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 8  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode E --h_reconstruct True
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 16 --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode E --h_reconstruct True
## ResNet-32 + ContSup[R1](decoder) => CIFAR-10 ##
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 2  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1 --h_reconstruct True
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 4  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1 --h_reconstruct True
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 8  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1 --h_reconstruct True
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 16 --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1 --h_reconstruct True
## ResNet-32 + ContSup[R1E](decoder) => CIFAR-10 ##
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 2  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E --h_reconstruct True
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 4  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E --h_reconstruct True
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 8  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E --h_reconstruct True
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 16 --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E --h_reconstruct True

## ResNet-32 + ContSup[E](contrast) => CIFAR-10 ##
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 2  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode E
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 4  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode E
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 8  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode E
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 16 --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode E
## ResNet-32 + ContSup[R1](contrast) => CIFAR-10 ##
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 2  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 4  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 8  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 16 --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1
## ResNet-32 + ContSup[R1E](contrast) => CIFAR-10 ##
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 2  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 4  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 8  --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 16 --local_loss_mode contrast --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1E



#########################################################
###### Table 4: Results with ResNet-32 on CIFAR-10 ######
#########################################################

## ResNet-32 + baseline[R0] => CIFAR-10 ##

CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 8  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R0
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 16 --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R0

## ResNet-32 + ContSup[E] => CIFAR-10 ##

CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 8  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode E
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 16 --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode E

## ResNet-32 + ContSup[R1] => CIFAR-10 ##

CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 8  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 16 --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R1

## ResNet-32 + ContSup[R2] => CIFAR-10 ##

CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 8  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R2
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 16 --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R2

## ResNet-32 + ContSup[R4] => CIFAR-10 ##

CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 8  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R4
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 16 --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R4

## ResNet-32 + ContSup[R8] => CIFAR-10 #

CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 8  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R8
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 16 --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R8

## ResNet-32 + ContSup[R16] => CIFAR-10 ##

CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 16 --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128  --aux_net_config 1c2f  --context_mode R16
