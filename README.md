
# SepReformer for Speech Separation [NeurIPS 2024]


This is the official implementation of “Separate and Reconstruct: Asymmetric Encoder-Decoder for Speech Separation” accepted in NeurIPS 2024 [Paper Link(Arxiv)](https://arxiv.org/abs/2406.05983)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/separate-and-reconstruct-asymmetric-encoder/speech-separation-on-wsj0-2mix)](https://paperswithcode.com/sota/speech-separation-on-wsj0-2mix?p=separate-and-reconstruct-asymmetric-encoder)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/separate-and-reconstruct-asymmetric-encoder/speech-separation-on-wham)](https://paperswithcode.com/sota/speech-separation-on-wham?p=separate-and-reconstruct-asymmetric-encoder)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/separate-and-reconstruct-asymmetric-encoder/speech-separation-on-whamr)](https://paperswithcode.com/sota/speech-separation-on-whamr?p=separate-and-reconstruct-asymmetric-encoder)




## News
🔥 October, 2024: We have uploaded the pre-trained models of our SepReformer-B for WSJ0-2MIX in `models/SepReformer_Base_WSJ0/log/scratch_weight` folder! You can directly test the model using the inference command below.

🔥 September 2024, Paper accepted at NeurIPS 2024 🎉.


## Todo
We are planning to release the other cases especially for partially or fully overlapped, noisy-reverberant mixture with 16k of sampling rates for practical application within this year.


![Untitled](data/figure/SepReformer_Architecture.png)

We  propose SepReformer, a novel approach to speech separation using an asymmetric encoder-decoder network named SepReformer. 

Demo Pages: [Sample Results of speech separation by SepReformer](https://fordemopage.github.io/SepReformer/)

### Requirement

- python 3.10
- torch 2.1.2
- torchaudio 2.1.2
- pyyaml 6.0.1
- ptflops
- mir_eval


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
    

### Training Curve
- For SepReformer-B with WSJ-2MIX, the training and validation curve is as follows:
![Untitled](data/figure/Training_Curve.png)

<br />
<br />

![Untitled](data/figure/Result_table.png)

![Untitled](data/figure/SISNRvsMACs.png)

### Citation

If you find this repository helpful, please consider citing:
```
@misc{shin2024separate,
      title={Separate and Reconstruct: Asymmetric Encoder-Decoder for Speech Separation}, 
      author={Ui-Hyeop Shin and Sangyoun Lee and Taehan Kim and Hyung-Min Park},
      year={2024},
      eprint={2406.05983},
      archivePrefix={arXiv},
}
```

## TODO
- [ ] To add the pretrained model.
