
## Table of Contents

- [Background](#background)
- [Installation](#installation)
- [Tutorial](#tutorial)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [License](#license)

## Background
The whole-slide images (WSI) examination of skin biopsy is the golden standard for pathological diagnosis of most skin diseases. While most studies focus on the classification tasks, an interpretable computational framework is lacking for WSI analysis. To this end, we developed PathoEye for WSI analysis in dermatology, which integrates epidermis-guided sampling, deep learning and radiomics. The established classification model using PathoEye performed better than the existing state-of-the-art methods in discriminating the young and aged skin. Moreover, PathoEye performs comparably with the existing methods in the binary classification of healthy and diseased skin while performing better in multi-classification tasks. 

## Install
Make sure you have installed all the package that were list in requirements.txt
```
conda create -n PathoEye python==3.8
pip install -r requirements.txt
conda activate PathoEye
```

## Step by step tutorial

### Testing dataset
This data can be downloaded from https://gtexportal.org/home/histologyPage




### Epidermis extraction
The following example assumes that the whole slide images (WSIs) data is organized in well known standard formats (such as .svs, .ndpi, .tiff etc.) and stored in a folder named DATA_DIRECTORY.
```
    DATA_DIRECTORY/
        ├──slide_1.svs
        ├──slide_1.svs
        ├──slide_2.svs
        ├──slide_3.svs
        └── ...
        
```
You can run epidermis_extract.py to extract epidermis of each WSI in the DATA_DIRECTORY as following. 
```sh
python epidermis_extract.py -data_dir /DATA_DIRECTORY -save_path ./save_dir
```
The epidermis layer will be saved in ./save_dir. 
```
    save_dir/
        ├──slide_1
            ├──1.png
            ├──2.png
            ├──3.png
            ├──...
            └──image_info.csv
        ├──slide_2
            ├──1.png
            ├──2.png
            ├──3.png
            ├──...
            └──image_info.csv
        ├──slide_3
            ├──1.png
            ├──2.png
            ├──3.png
            ├──...
            └──image_info.csv
        └── ...
```
Below is an example table for image_info.csv. 

| index | x |    y | screenshot_level | image_width| image_height|
|:------|     :---:      |-----:|           ---: |           ---: |           ---: |
| 1     |  32400    | 1120 | 1     |  900   |  900 |
| 2     |  36000    | 477  | 1     |  900   |  900 |
| 3     |  14400    | 4456 | 1     |  900   |  900 |

This table includes 6 columns. 
  - `index`: The image file name, corresponding to x.png.
  - `x`: The x-coordinate location to crop the image x.png in the WSI.
  - `y`: The y-coordinate location to crop the image x.png in the WSI.
  - `screenshot_level`: The zoom level of the image.
  - `image_width`: The width of the image.
  - `image_height`: The height of the image.

### Epidermis thickness and variance of rete ridge length calculation
You can apply thickness.py to calculate the thickness and the variance of rete ridge of each image in the DATA_DIRECTORY. The output will be saved in the save_dir directory.



```sh
python thickness.py -data_dir /DATA_DIRECTORY -save_path ./save_dir
```

### Epidermis-guided patch sampling
As described in the last step, the input images should be organized in the right file format and directory. Then, you can applied create_patches.py to segment images of the whole-slide images (WSIs).  
The organized WSIs must be stored under a folder named TRAIN_DIRECTORY(train dataset) or VAL_DIRECTORY (validation dataset). 
```
    TRAIN_DIRECTORY/
        ├── class_1
            ├──slide_1.svs
            ├──slide_2.svs
            ├──slide_3.svs
            └── ...
        ├── class_2
            ├──slide_1.svs
            ├──slide_2.svs
            ├──slide_3.svs
            └── ...
        ├── class_3
            ├──slide_1.svs
            ├──slide_2.svs
            ├──slide_3.svs
            └── ...
        └── ...

    VAL_DIRECTORY/
        ├── class_1
            ├──slide_1.svs
            ├──slide_2.svs
            ├──slide_3.svs
            └── ...
        ├── class_2
            ├──slide_1.svs
            ├──slide_2.svs
            ├──slide_3.svs
            └── ...
        ├── class_3
            ├──slide_1.svs
            ├──slide_2.svs
            ├──slide_3.svs
            └── ...
        └── ...
```
```sh
python create_patches.py -input_path /TRAIN_DIRECTORY -save_path /TRAIN_DATASET -device cuda:0
python create_patches.py -input_path /VAL_DIRECTORY -save_path /VAL_DATASET -device cuda:0
```
The above commands produce the following result for each slide:
```
    DATASET/
        ├── class_1
            ├──slide_1
                ├──data
                    ├──1.png
                    ├──2.png
                    └──...
                ├──mask_target.png
                └──raw_img.png
            ├──slide_2
                ├──data
                    ├──1.png
                    ├──2.png
                    └──...
                ├──mask_target.png
                └──raw_img.png
            └── ...
        ├── class_2
            ├──slide_1
                ├──data
                    ├──1.png
                    ├──2.png
                    └──...
                ├──mask_target.png
                └──raw_img.png
            ├──slide_2
                ├──data
                    ├──1.png
                    ├──2.png
                    └──...
                ├──mask_target.png
                └──raw_img.png
            └── ...
        └── ...
```

### DCNN training and classification
Next, you can train your own model using the segmented images (the result of the last step) as input. The VAL_DATASET folder will be generated by create_patches.py.
```sh
python train.py -train_path /TRAIN_DATASET -val_path /VAL_DATASET -save_path ./MODEL_SAVEPATH
```

### Classification testing
Subsequently, you can apply the trained model to perform binary classification on a new dataset organized in the right format and directory.
```sh
python inference.py -input_path /SLIDER.SVS -model_path /MODEL_SAVEPATH -save_path ./RESULT
```

## Please cite

Lin Y., Lin F., Zhang Y. et al. PathoEye: a deep learning framework for histopathological image analysis of skin tissue. Submitted.

## Maintainer

Any questions, please contact [@Yusen Lin](https://github.com/lysovosyl)

## Contributors

Thank you for the helps from Dr. Jiajian Zhou, Dr. Yongjun Zhang, Dr. Feiyan Lin and Miss Jiayu Wen.

## License

[MIT](LICENSE) © Yusen Lin
