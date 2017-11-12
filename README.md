md5sum: 15d0222e9737c3f558fad2e4d63f48d2
[snapshot](https://drive.google.com/uc?export=download&id=1T13xwXTLzxEHN_RF0i_0cvsetxX8H5vs)

## Getting Started
In order to reproduce results presented in Table 1 of the original [paper](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w17/Laskar_Camera_Relocalization_by_ICCV_2017_paper.pdf), please do the following:
* Download the 7-Scenes dataset from [here](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)
* Resize all the images such the smaller dimension is 256 and keep the aspect ratio:
 
```find . -name "*.color.png" | xargs -I {} convert {} -resize "256^>" {}```

* Download trained weights from [here](https://drive.google.com/uc?export=download&id=1T13xwXTLzxEHN_RF0i_0cvsetxX8H5vs)

# How to cite
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
