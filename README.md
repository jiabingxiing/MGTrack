# MGTrack

The official PyTorch implementation of MGTrack: 

## Let's Get Started

- ### Preparation

  - Clone our repository to your local project directory.

  - Download the training datasets ([LaSOT](http://vision.cs.stonybrook.edu/~lasot/download.html), [TrackingNet](https://github.com/SilvioGiancola/TrackingNet-devkit), [GOT-10k](http://got-10k.aitestunion.com/downloads), [COCO2017](https://cocodataset.org/#download)) and testing datasets to your disk, the organized directory should look like: 

    ```
    --LaSOT/
    	|--airplane
    	|...
    	|--zebra
    --TrackingNet/
    	|--TRAIN_0
    	|...
    	|--TEST
    --GOT10k/
    	|--test
    	|--train
    	|--val
    --COCO/
    	|--annotations
    	|--images
    --NFS30/
    	|--anno
    	|--sequences
    --OTB100/
    	|--Basketball
    	|...
    	|--Woman
    --UAV123/
    	|--anno
    	|--data_seq
    ```

  - Edit the **PATH** in ```lib/test/evaluation/local.py``` and ```lib/train/adim/local.py``` to the proper absolute path.

- ### Installation

  We use conda to manage the environment.

  ```
  conda create --name mgtrack python=3.8
  conda activate mgtrack
  sudo apt-get install ninja-build
  sudo apt-get install libturbojpeg
  bash install.sh
  ```

- ### Training

  - Multiple GPU training by DDP (suppose you have 8 GPU)

    ```
    python tracking/train.py --mode multiple --nproc 4  --config baseline
    ```

  - Single GPU debugging (too slow, not recommended for training)

    ```
    python tracking/train.py
    ```

  - For GOT-10k evaluation, remember to set ```--config baseline_got```.

- ### Evaluation

  - Make sure you have prepared the trained model.

  - On large-scale benchmarks: 

    - LaSOT

      ```
      python tracking/test.py --dataset got10k_test --param baseline_mask
      python tracking/test.py --dataset lasot --param baseline --threads 4
      ```
     - TNL2K  
      python tracking/test.py --dataset tnl2k --param baseline_mask --threads 4


    - TrackingNet

      ```
      python tracking/test.py --dataset trackingnet --param baseline_convnext_center --thread 4
      python lib/test/utils/transform_trackingnet.py --tracker_name tts --cfg_name baseline_mask1
      ```

      Then upload ```test/tracking_results/aiatrack/baseline/trackingnet_submit.zip``` to the [online evaluation server](https://eval.ai/web/challenges/challenge-page/1805/overview).

    - GOT-10k

      ```
      python tracking/test.py --param baseline --dataset got10k_test  --threads 4
      python lib/test/utils/transform_got10k.py --tracker_name tts --cfg_name baseline
      ```

      Then upload ```test/tracking_results/aiatrack/baseline_got/got10k_submit.zip``` to the [online evaluation server](http://got-10k.aitestunion.com/submit_instructions).


## Acknowledgement

:heart::heart::heart:Our idea is implemented base on the following projects. We really appreciate their wonderful open-source work!

- [STARK](https://github.com/researchmm/Stark) [[related paper](https://arxiv.org/abs/2103.17154)]
- [PyTracking](https://github.com/visionml/pytracking) [[related paper](https://arxiv.org/abs/1811.07628)]


