# Thermal Infrared Image Colorization on Nighttime Driving Scenes with Top-down Guided Attention
Pytorch implementation of the paper "Thermal Infrared Image Colorization on Nighttime Driving Scenes with Top-down Guided Attention".

![tease](https://github.com/FuyaLuo/PearlGAN/blob/main/docs/Model.PNG)

### [Paper](https://ieeexplore.ieee.org/abstract/document/9703249)

## Abstract
>Benefitting from insensitivity to light and high penetration of foggy environments, infrared cameras are widely used for sensing in nighttime traffic scenes. However, the low contrast and lack of chromaticity of thermal infrared (TIR) images hinder the human interpretation and portability of high-level computer vision algorithms. Colorization to translate a nighttime TIR image into a daytime color (NTIR2DC) image may be a promising way to facilitate nighttime scene perception. Despite recent impressive advances in image translation, semantic encoding entanglement and geometric distortion in the NTIR2DC task remain under-addressed. Hence, we propose a toP-down attEntion And gRadient aLignment based generative adversarial network, referred to as PearlGAN. A top-down guided attention module and an elaborate attentional loss are first designed to reduce the semantic encoding ambiguity during translation. Then, a structured gradient alignment loss is introduced to encourage edge consistency between the translated and input images. In addition, pixel-level annotation is carried out on a subset of FLIR and KAIST datasets to evaluate the semantic preservation performance of multiple translation methods. Furthermore, a new metric is devised to evaluate the geometric consistency in the translation process. Extensive experiments demonstrate the superiority of the proposed PearlGAN over other image translation methods for the NTIR2DC task. 

## Prerequisites
* Python 3.6 
* Pytorch 1.1.0 and torchvision 0.3.0 
* TensorboardX
* visdom
* dominate
* pytorch-msssim
* CUDA 10.0.130, CuDNN 7.3, and Ubuntu 16.04.

## Data Preparation 
Download [FLIR](https://www.flir.co.uk/oem/adas/adas-dataset-form/) and [KAIST](https://soonminhwang.github.io/rgbt-ped-detection/data/). First, the corresponding training set and test set images are sampled according to the txt files in the `./img_list/` folder. Then, all images are first resized to 500x400, and then crop centrally to obtain images with a resolution of 360x288. Note that due to negligent checking by the authors, the test set images for the KAIST dataset only need to be center cropped to 360x288 without a resize step. Finally, place all images into the corresponding dataset folders. Domain A and domain B correspond to the daytime visible image and the nighttime TIR image, respectively. As an example, the corresponding folder structure for the FLIR dataset is:
 ```
mkdir FLIR_datasets
# The directory structure should be this:
FLIR_datasets
   ├── trainA (daytime RGB images)
       ├── FLIR_00002.png 
       └── ...
   ├── trainB (nighttime IR images)
       ├── FLIR_00135.png
       └── ...
   ├── testA (testing daytime RGB images)
       ├── FLIR_09112.png (The test image that you want)
       └── ... 
   ├── testB (testing nighttime IR images)
       ├── FLIR_08872.png (The test image that you want)
       └── ... 

mkdir FLIR_testsets
# The directory structure should be this:
FLIR_testsets
   ├── test0 (empty folder)
   ├── test1 (testing nighttime IR images)
       ├── FLIR_08863.png
       └── ...
```

We predict the edge maps of Nighttime TIR images and daytime color images using [MCI](https://drive.google.com/file/d/1Qf2wIyzr0J8nWSuc8d6bHyO2Mxzeuamv/view?usp=sharing) method and Canny edge detection method, respectively. Next, place all edge maps into the corresponding folders(e.g., `/FLIR_IR_edge_map/` and `/FLIR_Vis_edge_map/` for FLIR dataset).

## Inference Using Pretrained Model

<details>
  <summary>
    <b>1) FLIR</b>
  </summary>
  
Download and unzip the [pretrained model](https://drive.google.com/file/d/19L1OPaLaAFcRoO6g6Hxws5Z9wVtX1A_h/view?usp=sharing) and save it in `./checkpoints/FLIR_NTIR2DC/`. Place the test images of the FLIR dataset in `./FLIR_testsets/test1/`. Then run the command 
```bash
python test.py --phase test --serial_test --name FLIR_NTIR2DC --dataroot ./FLIR_testsets/ --n_domains 2 --which_epoch 80 --results_dir ./res_FLIR/ --loadSize 288 --no_flip --net_Gen_type gen_v1 --gpu_ids 0
```
</details>

<details>
  <summary>
    <b>2) KAIST</b>
  </summary>
  
Download and unzip the [pretrained model](https://drive.google.com/file/d/1kGkv2g5LwnkyCdN_Hoa9bcwElXIzy6dR/view?usp=sharing) and save it in `./checkpoints/KAIST_NTIR2DC/`. Place the test images of the FLIR dataset in `./KAIST_testsets/test1/`. Then run the command 
```bash
python test.py --phase test --serial_test --name KAIST_NTIR2DC --dataroot ./KAIST_testsets/ --n_domains 2 --which_epoch 120 --results_dir ./res_KAIST/ --loadSize 288 --no_flip --net_Gen_type gen_v1 --gpu_ids 0
```
</details>

## Training

To reproduce the performance, we recommend that users try multiple training sessions.
<details>
  <summary>
    <b>1) FLIR</b>
  </summary>
  
  Place the corresponding images in each subfolder of the folder `./FLIR_datasets/`. Then run the command
  ```bash
  bash ./train_FLIR.sh
  ```
</details>


<details>
  <summary>
    <b>2) KAIST</b>
  </summary>
  
  Place the corresponding images in each subfolder of the folder `./KAIST_datasets/`. Then run the command
   ```bash
   bash ./train_KAIST.sh
   ```

</details>

## Labeled Segmentation Masks
We annotated a subset of [FLIR](https://drive.google.com/file/d/1IeyNBkWQQY9-AaZebalJumNtPt5wv9zR/view?usp=sharing) and [KAIST](https://drive.google.com/file/d/1CQz6yZjxdVarHMcWFdxirTLpTdNu7_da/view?usp=sharing) datasets with pixel-level category labels, which may catalyze research on the colorization and semantic segmentation of nighttime TIR images. The indexing of categories is consistent with the annotation of the Cityscape dataset.

![Labeled Masks](https://github.com/FuyaLuo/PearlGAN/blob/main/docs/Masks.PNG)

## Evaluation
<details>
  <summary>
    <b>1) Semantic segmenation</b>
  </summary>
  
   Download the code for the domain adaptation semantic segmentation model [MRNet](https://github.com/layumi/Seg-Uncertainty) and then follow the instructions to install it. Next, download our pre-trained models and associated files on [FLIR](https://drive.google.com/file/d/1ZKGHJgstg9KL9wMFhM1z4zZJM6J3tjS5/view?usp=sharing) and [KAIST](https://drive.google.com/file/d/1jpNgF--yHoyK2IbVEB5LRU9F72dVuceB/view?usp=sharing) datasets using Cityscape datasets. Once the unzip is complete, place all files in the `/Seg-Uncertainty-master/` folder. Note that the files `FLIR_dataset.py` and `KAIST_dataset.py` should be placed in the directory `/Seg-Uncertainty-master/dataset/`. For the evaluation on FLIR dataset, run the command
   ```bash
   python evaluate_FLIR_class9.py --data_dir /Your_FLIR_Results_Path/
   ```
   And for the evaluation on KAIST dataset, run the command
   ```bash
   python evaluate_KAIST_class9.py --data_dir /Your_KAIST_Results_Path/
   ```
   
</details>

<details>
  <summary>
    <b>2) Object detection</b>
  </summary>
  
  Download the code for [YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4), then follow the instructions to install it. Since the weights for YOLOv4 were subsequently updated, we shared the [weights](https://drive.google.com/file/d/1llLoMUHh9MJRMyiPclTfVDTWUrIiieJF/view?usp=sharing) we utilized for testing mAP at that time. Note that the weights shared are for non-commercial research only. Next, download the YOLOv4 detection txt file we transformed from the [FLIR](https://drive.google.com/file/d/1BCWFAylBWfLXkc27tdZ4r20hbkmNyEtg/view?usp=sharing) and [KAIST](https://drive.google.com/file/d/15EODc69Ivm-c4wsscBr9UrwnmY0YV3Eu/view?usp=sharing) datasets. Once the unzip is complete, place all files in the `/PyTorch_YOLOv4-master/` folder. Note that the files `FLIR.yaml`, `FLIR_imglist.txt`, `KAIST.yaml` and `KAIST_imglist.txt` should be placed in the directory `/PyTorch_YOLOv4-master/data/`. And the file `yolov4.weights` should be placed in the directory `/PyTorch_YOLOv4-master/weights/`. Then, the translation results of FLIR and KAIST should be placed inside the `/PyTorch_YOLOv4-master/FLIR_datasets/images/` and `/PyTorch_YOLOv4-master/KAIST_datasets/images/` directories respectively. For the evaluation on FLIR dataset, run the command
  ```bash
  python test.py --img 640 --conf 0.001 --batch 32 --device 0 --data FLIR.yaml --cfg cfg/yolov4.cfg --weights weights/yolov4.weights --verbose
  ```
  And for the evaluation on KAIST dataset, run the command
   ```bash
   python test.py --img 640 --conf 0.001 --batch 32 --device 0 --data KAIST.yaml --cfg cfg/yolov4.cfg --weights weights/yolov4.weights --verbose
   ```
       
</details>


<details>
  <summary>
    <b>3) Edge consistency</b>
  </summary>
  
   Change the values of the variables `night_IR_folder`, `file_cell_list{1, 2}`, `des_txt_path` in file `./APCE_eval/batch_eval_CE_FLIR_single.m` to `your NTIR directory`, `your translation result directory` and `your txt result storage directory` respectively. Next, run file `batch_eval_CE_FLIR_single.m` to evaluate the APCE of the results on the FLIR dataset. Similar to evaluate the results on the KAIST dataset by running file `batch_eval_CE_KAIST_single.m`. Be careful to keep the file name of the resulting image the same as the file name of the original IR image.

    
</details>

![Edge Consistency Comparison](https://github.com/FuyaLuo/PearlGAN/blob/main/docs/KAIST_APCE_example.PNG)

## Downloading files using Baidu Cloud Drive
If the above Google Drive link is not available, you can try to download the relevant code and files through the [Baidu cloud link](https://pan.baidu.com/s/1ojaqDf6dV_XYrsOqi1NNAg), extraction code: ir2d.

## Citation
If you like our work and use the code or models for your research, please cite our work as follows.
```
@article{luo2022thermal,
  title={Thermal infrared image colorization for nighttime driving scenes with top-down guided attention},
  author={Luo, Fuya and Li, Yunhan and Zeng, Guang and Peng, Peng and Wang, Gang and Li, Yongjie},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2022},
  publisher={IEEE}
}
```

## License

The codes and the pretrained model in this repository are under the BSD 2-Clause "Simplified" license as specified by the LICENSE file. 

## Acknowledgments
This code is heavily borrowed from [ToDayGAN](https://github.com/AAnoosheh/ToDayGAN).  
Spectral Normalization code is borrowed from [BigGAN-PyTorch](https://github.com/ajbrock/BigGAN-PyTorch/blob/master/layers.py).  
We thank [Lucky0775](https://github.com/Lucky0775) for his questions about the object detection results.  
We thank [yupan233](https://github.com/yupan233) for his questions about the preprocessing step for the KAIST dataset.
