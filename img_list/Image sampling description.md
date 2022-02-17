
# Image sampling

Based on the correspondence between thermal infrared images and RGB images, we manually divide the dataset into daytime and nighttime parts according to the lighting conditions of RGB images.

## FLIR
After removing some noisy images (e.g., all-white images), we finally obtained 5447 daytime color (DC) images and 2899 nighttime thermal infrared (NTIR) images for the training of the NTIR2DC task. Similarly, we were able to collect 490 NTIR images in the validation set for model evaluation.

## KAIST
The KAIST dataset contains video data from three types of traffic scenes: campus, street and suburban, which are divided into 12 folders (i.e., from `Set00` to `Set11`) in total. To cover different types of scenes, we started to select DC images from `Set00` to `Set02` and NTIR images from `Set03` to `Set05` as the training set, while NTIR images from `Set09` to `Set11` as the test set. Since `Set10` contains a long video of following a truck, which severely deviates from the data distribution of the training set. Therefore, to ensure the similarity between the data distributions of the training and test sets, we finally chose to discard `Set10` and divide the second video in `Set04` into the test set. To reduce the similarity between frame images and the gap in the number of cross-domain data, we sampled DC and NTIR video data at intervals of 20 and 10 frames, respectively, and finally obtained 1674 DC images and 1359 NTIR images as training set samples.

For the collection of test sets, to reduce the similarity of test images and the burden of repetitive annotation, we first sampled NTIR images from test video frames at 20-frame intervals, and then randomly selected 500 images for annotation and test.
