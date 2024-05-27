import sys
sys.path.append('../')

import torch
import warnings
warnings.filterwarnings('ignore')

from utils.decorators import *
from .modules.module import *


@logger_wraps()
class Model(torch.nn.Module):
    def __init__(self, 
                 num_stages: int, 
                 num_spks: int, 
                 module_audio_enc: dict, 
                 module_feature_projector: dict, 
                 module_separator: dict, 
                 module_output_layer: dict, 
                 module_audio_dec: dict):
        super().__init__()
        self.num_stages = num_stages
        self.num_spks = num_spks
        self.audio_encoder = AudioEncoder(**module_audio_enc)
        self.feature_projector = FeatureProjector(**module_feature_projector)
        self.separator = Separator(**module_separator)
        self.out_layer = OutputLayer(**module_output_layer)
        self.audio_decoder = AudioDecoder(**module_audio_dec)
        
        # Aux_loss
        self.out_layer_bn = torch.nn.ModuleList([])
        self.decoder_bn = torch.nn.ModuleList([])
        for _ in range(self.num_stages):
            self.out_layer_bn.append(OutputLayer(**module_output_layer, masking=True))
            self.decoder_bn.append(AudioDecoder(**module_audio_dec))
        
    def forward(self, x):
        encoder_output = self.audio_encoder(x)
        projected_feature = self.feature_projector(encoder_output)
        last_stage_output, each_stage_outputs = self.separator(projected_feature)
        out_layer_output = self.out_layer(last_stage_output, encoder_output)
        each_spk_output = [out_layer_output[idx] for idx in range(self.num_spks)]
        audio = [self.audio_decoder(each_spk_output[idx]) for idx in range(self.num_spks)]
        
        # Aux_loss
        audio_aux = []
        for idx, each_stage_output in enumerate(each_stage_outputs):
            each_stage_output = self.out_layer_bn[idx](torch.nn.functional.upsample(each_stage_output, encoder_output.shape[-1]), encoder_output)
            out_aux = [each_stage_output[jdx] for jdx in range(self.num_spks)]
            audio_aux.append([self.decoder_bn[idx](out_aux[jdx])[...,:x.shape[-1]] for jdx in range(self.num_spks)])
            
        return audio, audio_aux