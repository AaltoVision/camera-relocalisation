
Source code (Torch, MATLAB), models, and dataset for _Camera Relocalization by Computing Pairwise Relative Poses Using Convolutional Neural Network_ [link](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w17/Laskar_Camera_Relocalization_by_ICCV_2017_paper.pdf)

## Getting Started
In order to reproduce results presented in Table 1 of the original paper, please do the following:
* Download the 7-Scenes dataset from [here.](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)
* Resize all the images such the smaller dimension is 256 and keep the aspect ratio:
```
find . -name "*.color.png" | xargs -I {} convert {} -resize "256^>" {}
```
* Download trained weights from [here](https://drive.google.com/uc?export=download&id=1T13xwXTLzxEHN_RF0i_0cvsetxX8H5vs) (md5sum: 15d0222e9737c3f558fad2e4d63f48d2).
* From ```cnn_part``` folder run:
```
th main.lua -weights <path/to/downloaded_weights/model_snapshot_7scenes.t7> -dataset_src_path </path/to/7Scenes> -do_evaluation
```
By default, calculated features would be saved to ```cnn_part/results/results.bin```
* To measure localisation performance run ```matlab filter_pose.m```

## University Dataset

## How to cite
If you use this software in your own research, please cite our publication:

```
@InProceedings{LMKK2017ICCVW,
    author = {Laskar, Zakaria and Melekhov, Iaroslav and Kalia, Surya and Kannala, Juho},
    title = {Camera Relocalization by Computing Pairwise Relative Poses Using Convolutional Neural Network},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {Oct},
    year = {2017}
}
```
