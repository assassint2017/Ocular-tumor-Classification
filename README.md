# Ocular-tumor-Classification

using deep learning to identify ocular tumor types.
there are two type of tumor:yz and lbl

The data we use comes from a local partner hospital. include 84 lbl and 73 yz.Each mri data includes three sequences: T1, T2 and T1C, we use T1C and T2 train a small DenseNet to classifiy the tumor.for each training and val data, we first manually segment the most visible area of the lesion to provide a ROI, and then feed this area to the ConVnet.since we don't have enough data, to test our algorithm, we used five-fold cross-validation.Here are the results we get take average on five folds:

|mean accuracy|mean precision|mean recall|mean F1_score|
|:--:|:--:|:--:|:--:|
|0.780|0.752|0.886|0.812|
