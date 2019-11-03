# DeepLabv3plus-TensorFlow
- Official paper: Chen, L.-C. et al., Encoder-Decoder with atrous separable convolution for semantic image segmentation, Proceedings of the European conference on computer vision (ECCV), 2018.
- Paper link: http://openaccess.thecvf.com/content_ECCV_2018/html/Liang-Chieh_Chen_Encoder-Decoder_with_Atrous_ECCV_2018_paper.html
- A powerful deep learning architecture for image semantic segmentation
- CNN-based encoder-decoder with atrous separable convolution
## In this code
- DeepLabv3+ == Modified aligned Xception + ASPP + U-Net style decoder
- In a training phase, random cropping and random scaling were employed and then resized to the fixed training image size.
- tf.\_\_version\_\_ == '1.12.0' ~ '1.14.0'
- Only CPU: set FLAGS.gpu_num to 0 in main.py.
- The number of GPUs > 2: mannually allocate them.
- Inputs shape: (N, H, W, C) (0~1)
- Ground truths shape: (N, H, W, num_class) (0 or 1)
## Author
Sehyeok Oh @shoh4486
## Author's application
- Semantic segmentation of self-piercing riveting (SPR) optical microscopic images (mIOU: 98.49%).
- (to be updated)
