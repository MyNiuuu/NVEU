# NIR-assisted Low-light Video Enhancement Using Unpaired 24-hour Data

In this repository, we provide source codes and pretrained checkpoint for run the test demo. The full dataset and training codes will be available in the future. Please submit an issue or contact muyao.niu@gmail.com if you are in a hurry for these.

## Environment Setup
```
conda create -n nir python==3.8
conda activate nir
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html
pip install opencv-python
cd package_core
python setup.py install
cd ..
pip install timm
```

## Download pretrained checkpoint
Download checkpoints from [here](https://drive.google.com/file/d/12wxxUenS4MYIco5kswsbV-vCDRVRh_qn/view?usp=drive_link), and put it into `./`.

## Run Test Script
```
chmod 777 test.sh
./test.sh
```
The results will be saved to `./test_results`.
