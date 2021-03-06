# DeepLabv3plus-TensorFlow
- Official paper: Chen, L.-C. et al., Encoder-Decoder with atrous separable convolution for semantic image segmentation, Proceedings of the European conference on computer vision (ECCV), 2018.
- Paper link: http://openaccess.thecvf.com/content_ECCV_2018/html/Liang-Chieh_Chen_Encoder-Decoder_with_Atrous_ECCV_2018_paper.html
- A powerful deep learning architecture for image semantic segmentation
- CNN-based encoder-decoder with atrous separable convolution
## In this code
- DeepLabv3+ == Modified aligned Xception + ASPP + U-Net style decoder
- Augmentation
  - In the training phase, random scaling (only enlargement), random cropping and random flipping were employed to the input training data of shape (H, W) = (H_orig, W_orig).
  - The augmented training data was then resized to the fixed size of (H, W) = (H_train, W_train), which should be set through the argument parser.
  - Thus, the real image size while training is smaller than the original image size (H_train < H_orig, W_train < W_orig). 
  - Moderately set H_train and W_train in argument parser if not to keep aspect ratio in random scaling. To keep it, H_train:W_train must be equal to H_orig:W_orig.
  
- tf.\_\_version\_\_ == '1.12.0' ~ '1.15.0' (1.15 tested)
- The number of GPUs > 2: mannually allocate them.
- **Inputs shape: (N, H, W, C) (0~1)**
- **Ground truths shape: (N, H, W, num_class) (0 or 1)**
## Run example
- training mode: 
```
$ python main.py --trial_num=1 --C_in=3 --num_class=4 --H_train=250 --W_train=350 --n_aug=100 --train=True --start_epoch=0 --end_epoch=200 --gpu_alloc=1,2
```
- testing mode: 
```
$ python main.py --trial_num=2 --train=False --restore=True --restore_trial_num=1 --restore_sess_num=199 --eval_with_test_acc=True --output_stride_testing=8 --gpu_alloc=1,2
```
- Add other FLAGS options if necessary
## Author
Sehyeok Oh @shoh4486
## Author's application
- Semantic segmentation of self-piercing riveting (SPR) optical microscopic images (mIOU: 98.49%).
- (to be updated)
