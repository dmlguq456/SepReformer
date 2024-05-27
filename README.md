
# SepReformer


Official implementation from the following paper:

“Separate and Reconstruct: Asymmetric Encoder-Decoder for Speech Separation”

![Untitled](data/figure/SepReformer_Architecture.png)

We  propose SepReformer, a novel approach to speech separation using an asymmetric encoder-decoder network named SepReformer. 

Demo Pages: [Sample Results of speech separation by SepReformer](https://fordemopage.github.io/SepReformer/)

### Requirement

- python 3.10
- torch 2.1.2
- torchaudio 2.1.2
- pyyaml 6.0.1
- ptflops
- wandb


### Features

- You can log the training process by **wandb** as well as tensorboard.
- Support **dynamic mixing (DM)** in training

### Data Preparation

- For training or evaluation, you need dataset and scp file
    1. Prepare dataset for speech separation (eg. WSJ0-2mix)
    2. create scp file using data/crate_scp/*.py

### Training

- If you want to train the network, you can simply trying by
    - set the scp file in ‘models/SepReformer_Base_WSJ0/configs.yaml’
    - run training as
        
        ```bash
        python run.py --model SepReformer_Base_WSJ0 --engine-mode train
        ```
        

### Inference

- Simply evaluating a model without saving output as audio files
    
    ```bash
    python run.py --model SepReformer_Base_WSJ0 --engine-mode test
    ```
    

- Evaluating with output wav files saved
    
    ```bash
    python run.py --model SepReformer_Base_WSJ0 --engine-mode test_wav --out_wav_dir '/your/save/directoy[optional]'
    ```
    

![Untitled](data/figure/Result_table.png)

![Untitled](data/figure/SISNRvsMACs.png)

### Citation

If you find this repository helpful, please consider citing:
