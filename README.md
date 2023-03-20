# PRIDE (Pseudo-label Refinement with Intensive Knowledge transfer and DisEntangled attention)

This repository contains the official Pytorch implementation of our paper: **Intensive Knowledge Transfer for Weakly-Supervised Video Anomaly Detection**.

- Authors: Jash Dalvi, Gunjan Dhanuka, Ali Dabouei, Min Xu
- Achieves state-of-the-art results on UCF-Crime and ShanghaiTech datasets.

## Setting Up
- Clone the repository and navigate into the repo.

    ```
    git clone https://github.com/GunjanDhanuka/PRIDE
    cd PRIDE/
    ```

- Please download the required files from the drive link: [Drive Link](https://drive.google.com/drive/folders/1vUirYygnRdiEyOYXsUjdNOPrXL3lrMn4?usp=share_link), extract them and place in the `data` folder.

- The final directory structure should look like - 
    ```
    |--PRIDE/
        |-- config/
        |-- data/
            |-- I3D
                |-- all_rgbs/
                |-- all_flows/
            |-- S3D/
            |-- VideoSwin/
        |-- DataLoaders/
        |-- files/
        |-- Losses/
        |-- Models/
        |-- Utils/
        ...
    ```

- Change the file locations in `config/config.yaml` file to the absolute path of the folders on your device.

- Create a new virtual environment using your preferred method and install the required dependencies from the `requirements.txt` file. Note that we recommend installing PyTorch using the commands given on the PyTorch website, and then installing the rest of the packages using pip.

