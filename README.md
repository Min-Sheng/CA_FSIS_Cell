# Class-agnostic Few-shot Instance Segmentation of Digital Pathological Images

![Image](images/method.png)

This project is a pure pytorch implementation of *Class-agnostic Few-shot Instance Segmentation of Digital Pathological Images*. A majority of the code is modified from [roytseng-tw/Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch).

## Getting Started
Clone the repo:

```bash
git clone https://github.com/Min-Sheng/CA_FSIS_Cell.git
```

## Requirements

Tested under python3.

- python packages
  - pytorch>=0.3.1
  - torchvision>=0.2.0
  - cython
  - matplotlib
  - numpy
  - scipy
  - opencv
  - pyyaml
  - packaging
  - [pycocotools](https://github.com/cocodataset/cocoapi)  — for COCO dataset, also available from pip.
  - tensorboardX  — for logging the losses in Tensorboard.
- An NVIDAI GPU and CUDA 8.0 or higher. Some operations only have gpu implementation.

## Compilation

Compile the cuda dependencies using following simple commands:

```bash
cd lib  # please change to this directory
sh make.sh
```

It will compile all the modules you need, including NMS, ROI_Pooing, ROI_Crop and ROI_Align. (Actually gpu nms is never used ...).

## Data Preparation

Create a data folder under the repo,

```bash
cd {repo_root}
mkdir data
```

- **FIS-Cell**:
  Download the FIS-Cell(Few-shot Instance Segmentation for Cell images) dataset.

  Feel free to put the dataset at any place you want, and then soft link the dataset under the `data/` folder:

   ```bash
   ln -s path/to/FIS-Cell data/fss_cell
   ```

  Recommend to put the images on a SSD for possible better training performance.

## Pretrained Model

We use ResNet50 as the pretrained model in our experiments. This pretrained model is from [timy90022/One-Shot-Object-Detection](https://github.com/timy90022/One-Shot-Object-Detection) and available at:

* ResNet50: [Google Drive](https://drive.google.com/file/d/1SL9DDezW-neieqxWyNlheNefwgLanEoV/view?usp=sharing)

Download and unzip them into the `{repo_root}/data/`.

## Training

Use the environment variable `CUDA_VISIBLE_DEVICES` to control which GPUs to use.

In FIS-Cell dataset, we split it into 5 class splits. It will train and test different class. Just to adjust `--g (1~5)`.

If you want to train parts of the dataset, try to modify `--seen`:

- 1: only see training class (for training).
- 2: only see testing class (for testing).
- 3: see both training class and testing class.

### Adapative config adjustment

### Setting

- batch_size:            `NUM_GPUS` x `TRAIN.IMS_PER_BATCH`  
- effective_batch_size:  batch_size x `iter_size`  
- change of somethining: `new value of something / old value of something`

Following config options will be adjusted **automatically** according to actual training setups: 
1. number of GPUs `NUM_GPUS`
2. batch size per GPU `TRAIN.IMS_PER_BATCH`
3. update period `iter_size`

- `SOLVER.BASE_LR`: adjust directly propotional to the change of batch_size.
- `SOLVER.STEPS`, `SOLVER.MAX_ITER`: adjust inversely propotional to the change of effective_batch_size.

### Train from scratch
Take mask-rcnn with res50 backbone for example.
```
python tools/train_net_step.py --dataset coco2017 --cfg configs/baselines/e2e_mask_rcnn_R-50-C4.yml --use_tfboard --bs {batch_size} --nw {num_workers}
```

Use `--bs` to overwrite the default batch size to a proper value that fits into your GPUs. Simliar for `--nw`, number of data loader threads defaults to 4 in config.py.

Specify `—-use_tfboard` to log the losses on Tensorboard.

**NOTE**: use `--dataset keypoints_coco2017` when training for keypoint-rcnn.

### The use of `--iter_size`
As in Caffe, update network once (`optimizer.step()`) every `iter_size` iterations (forward + backward). This way to have a larger effective batch size for training. Notice that, step count is only increased after network update.

```
python tools/train_net_step.py --dataset coco2017 --cfg configs/baselines/e2e_mask_rcnn_R-50-C4.yml --bs 4 --iter_size 4
```
`iter_size` defaults to 1.

### Finetune from a pretrained checkpoint
```
python tools/train_net_step.py ... --load_ckpt {path/to/the/checkpoint}
```
or using Detectron's checkpoint file
```
python tools/train_net_step.py ... --load_detectron {path/to/the/checkpoint}
```

### Resume training with the same dataset and batch size
```
python tools/train_net_step.py ... --load_ckpt {path/to/the/checkpoint} --resume
```
When resume the training, **step count** and **optimizer state** will also be restored from the checkpoint. For SGD optimizer, optimizer state contains the momentum for each trainable parameter.

**NOTE**: `--resume` is not yet supported for `--load_detectron`

### Set config options in command line
```
  python tools/train_net_step.py ... --no_save --set {config.name1} {value1} {config.name2} {value2} ...
```
- For Example, run for debugging.
  ```
  python tools/train_net_step.py ... --no_save --set DEBUG True
  ```
  Load less annotations to accelarate training progress. Add `--no_save` to avoid saving any checkpoint or logging.

### Show command line help messages
```
python train_net_step.py --help
```

### Two Training Scripts

In short, use `train_net_step.py`.

In `train_net_step.py`:
- `SOLVER.LR_POLICY: steps_with_decay` is supported.
- Training warm up in [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677) is supported.

(Deprecated) In `train_net.py` some config options have no effects and worth noticing:

 - `SOLVER.LR_POLICY`, `SOLVER.MAX_ITER`, `SOLVER.STEPS`,`SOLVER.LRS`:
  For now, the training policy is controlled by these command line arguments:

    - **`--epochs`**: How many epochs to train. One epoch means one travel through the whole training sets. Defaults to  6.
    - **`--lr_decay_epochs `**: Epochs to decay the learning rate on. Decay happens on the beginning of a epoch. Epoch is 0-indexed. Defaults to [4, 5].

   For more command line arguments, please refer to `python train_net.py --help`

- `SOLVER.WARM_UP_ITERS`, `SOLVER.WARM_UP_FACTOR`, `SOLVER.WARM_UP_METHOD`:
  Training warm up is not supported.

## Inference

### Evaluate the training results
For example, test mask-rcnn on coco2017 val set
```
python tools/test_net.py --dataset coco2017 --cfg config/baselines/e2e_mask_rcnn_R-50-FPN_1x.yaml --load_ckpt {path/to/your/checkpoint}
```
Use `--load_detectron` to load Detectron's checkpoint. If multiple gpus are available, add `--multi-gpu-testing`.

Specify a different output directry, use `--output_dir {...}`. Defaults to `{the/parent/dir/of/checkpoint}/test`

### Visualize the training results on images
```
python tools/infer_simple.py --dataset coco --cfg cfgs/baselines/e2e_mask_rcnn_R-50-C4.yml --load_ckpt {path/to/your/checkpoint} --image_dir {dir/of/input/images}  --output_dir {dir/to/save/visualizations}
```
`--output_dir` defaults to `infer_outputs`.

## Configuration Options

Architecture specific configuration files are put under [configs](configs/). The general configuration file is [lib/core/config.py](lib/core/config.py).

## Acknowledgments
Code is based on [roytseng-tw/Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch) and [timy90022/One-Shot-Object-Detection](https://github.com/timy90022/One-Shot-Object-Detection) and [oeway/pytorch-deform-conv](https://github.com/oeway/pytorch-deform-conv).
