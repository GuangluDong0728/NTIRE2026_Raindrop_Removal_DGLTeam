# [NTIRE2026] - [The Second Challenge on Day and Night Raindrop Removal for Dual-Focused Images] - [DGLTeam]

# Environment Prepare
You can refer to the environment preparation process of [BasicSR](https://github.com/XPixelGroup/BasicSR), which mainly includes the following two steps:

1. 

    ```bash
    pip install -r requirements.txt
    ```

2. 

    ```bash
    python setup.py develop
    ```

# Downloading Our Weights

1. **Download Pretrained Weights:**
   - Navigate to [this link](https://drive.google.com/file/d/1MNiD5IMLxIqpRseO1vJUUyZ_2Sj2oX4f/view?usp=sharing) to download our weights. 

2. **Save to `experiments` Directory:**
   - Once downloaded, place the weights into the `experiments` directory.
     
# Testing

To test our model, please open the `options/NTIRE2026/test_se_raindrop.yml` file and update the paths, and just run the command:

```bash
python basicsr/test.py -opt options/NTIRE2026/test_se_raindrop.yml
```
Then, please use rename2.py to rename the images.
Finally please use post_scene_fuse_blend.py and update the paths to get our final outputs.

# Factsheet and results on testset

1. **Download Factsheet:**
   - Navigate to [this link](https://drive.google.com/file/d/1xM5dwMoa56bfaiW_5pk5M30-V-PYFpe1/view?usp=drive_link) to download our Factsheet.
     
2. **Download results on testset:**
   - Navigate to [this link](https://drive.google.com/file/d/1NqgLXRCjmnEmCfEm9A3bVHG78hdib-3g/view?usp=sharing) to download our results on testset. 

# Acknowledgements

This project is built on source codes shared by [BasicSR](https://github.com/XPixelGroup/BasicSR).
