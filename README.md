
Source code (Torch, MATLAB), models, and dataset for _Camera Relocalization by Computing Pairwise Relative Poses Using Convolutional Neural Network_ [link.](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w17/Laskar_Camera_Relocalization_by_ICCV_2017_paper.pdf)

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
By default, calculated features would be saved to ```cnn_part/results/7scenes_res.bin```
* To measure localisation performance run ```matlab filter_pose.m```

scene|[PoseNet](https://github.com/alexgkendall/caffe-posenet)|LSTM-Pose [[paper]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Walch_Image-Based_Localization_Using_ICCV_2017_paper.pdf)|VidLoc [[paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Clark_VidLoc_A_Deep_CVPR_2017_paper.pdf)|Ours
:---|:---:|:---:|:---:|:---:
Chess|0.32m, 8.12deg|0.24m, 5.77deg|0.18m, N/A|0.13m, 6.46deg
Fire |0.47m, 14.4deg|0.34m, 11.9deg|0.26m, N/A|0.26m, 12.72deg
Heads|0.29m, 12.0deg|0.21m, 13.7deg|0.14m, N/A|0.14m, 12.34deg
Office|0.48m, 7.68deg|0.30m, 8.08deg|0.26m, N/A|0.21m, 7.35deg
Pumpkin|0.47m, 8.42deg|0.33m, 7.00deg|0.36m, N/A|0.24m, 6.35deg
Red Kitchen|0.59m, 8.64deg|0.37m, 8.83deg|0.31m, N/A|0.24m, 8.03deg
Stairs|0.47m, 13.8deg|0.40m, 13.7deg|0.26m, N/A|0.27m, 11.82deg
Average|**0.44m**, **10.4deg**|**0.31m**, **9.85deg**|**0.25m**, **N/A**|**0.21m**, **9.30deg**


## University Dataset
* The University dataset (~29Gb) is available [here](https://drive.google.com/uc?export=download&id=1BUpZDDcphmwtlgb2I9JrpMo9p_8CrJJX) (md5sum: 6f512e6c55006c3f6fa0bf3f75f93284).
* Resize images according to 7-Scenes dataset (keeping aspect ratio).
* Download trained weights from [here](https://drive.google.com/uc?export=download&id=1cUc8IQVUxBmku1wBODUM82td8eRibC2Y) (md5sum: 227caa217653b48ffdf27a9826b838e5).
* From ```cnn_part``` folder run:
```
th main.lua -dataset_name University -weights <path/to/downloaded_weights/model_snapshot_university.t7> -dataset_src_path </path/to/University> -results_filename ./results/university_res.bin -do_evaluation
```
* To measure localisation performance open ```filter_pose.m``` and change ```pred_file_id``` to a binary file with estimates, i.e ```cnn_part/results/university_res.bin```

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
