# Maskrcnn-implementation
It's a implementation of MaskRCNN by Tippy.

## Note
In my main training script **MN_MaskRCNN_train_byT.py**, I save the image, mask, class, and score in **.mat** file format during the detection process. Therefore, you can use **MN_metrics_byT.py** to read the .mat files and calculate the segmentation, classification, and mPQ metrics.

If you are interested in the mPQ metric, please refer to the following link: [CoNic Challenge 2022](https://conic-challenge.grand-challenge.org/Evaluation/)
