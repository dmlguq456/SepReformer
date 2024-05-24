% create_wav_2_speakers.m
%
% Create 2-speaker mixtures
% 
% This script assumes that WSJ0's wv1 sphere files have already
% been converted to wav files, using the original folder structure
% under wsj0/, e.g., 
% 11-1.1/wsj0/si_tr_s/01t/01to030v.wv1 is converted to wav and 
% stored in YOUR_PATH/wsj0/si_tr_s/01t/01to030v.wav, and
% 11-6.1/wsj0/si_dt_05/050/050a0501.wv1 is converted to wav and
% stored in YOUR_PATH/wsj0/si_dt_05/050/050a0501.wav.
% Relevant data from all disks are assumed merged under YOUR_PATH/wsj0/
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Copyright (C) 2016 Mitsubishi Electric Research Labs 
%                          (Jonathan Le Roux, John R. Hershey, Zhuo Chen)
%   Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% data_type = {'cv','tt'};
data_type = {'tr','cv','tt'};
wsj0root = '/home/nas/user/Uihyeop/DB/wsj0-mix/2speakers/wav8k/min'; % YOUR_PATH/, the folder containing wsj0/
output_dir16k='/home/nas/user/Uihyeop/DB/wsj0-mix/2speakers/wav8k/min_segment';
% output_dir8k='/home/nas/user/Uihyeop/DB/wsj0-mix/2speakers/wav8k/css_v3_segment';

min_max = 'min';
% min_max = {'min','max', 'css'};

current_dir = cd('.');
for i_type = 1:length(data_type)
    cd([wsj0root '/' data_type{i_type} '/mix/']); file_list = dir('*.wav'); cd(current_dir);
    % cd([wsj0root,'/',data_type(i_type), '/s2/']); file_list_s2 = dir('*.wav'); cd(current_dir);
    % cd([wsj0root,'/',data_type(i_type), '/mix/']); file_list_mix = dir('*.wav'); cd(current_dir);


    if ~exist([output_dir16k '/' data_type{i_type}],'dir')
        mkdir([output_dir16k '/' data_type{i_type}]);
    end
    status = mkdir([output_dir16k '/' data_type{i_type} '/s1/']); %#ok<NASGU>
    status = mkdir([output_dir16k '/' data_type{i_type} '/s2/']); %#ok<NASGU>
    status = mkdir([output_dir16k '/' data_type{i_type} '/mix/']);

    fprintf(1,'%s\n',[min_max '_' data_type{i_type}]);
    for file = 1:size(file_list,1)
        file_name = file_list(file).name(1:end-4);
        [s1_16k, fs] = audioread([wsj0root, '/', data_type{i_type},'/s1/', file_name, '.wav']);
        s2_16k       = audioread([wsj0root, '/', data_type{i_type},'/s2/', file_name,'.wav']);
        mix_16k       = audioread([wsj0root, '/', data_type{i_type},'/mix/', file_name, '.wav']);

        s1_16k = int16(round((2^15)*s1_16k));
        s2_16k = int16(round((2^15)*s2_16k));
        mix_16k = int16(round((2^15)*mix_16k));
        mix_16k_length = length(mix_16k);
        cut_len = 4*fs;
        max_len = 6*fs;
        if length(mix_16k) <= max_len
            audiowrite([output_dir16k '/' data_type{i_type} '/s1/' file_name '.wav'],s1_16k,fs);
            audiowrite([output_dir16k '/' data_type{i_type} '/s2/' file_name '.wav'],s2_16k,fs);
            audiowrite([output_dir16k '/' data_type{i_type} '/mix/' file_name '.wav'],mix_16k,fs);
        else
            num_seg = fix(length(mix_16k)/(cut_len));
            for seg_i = 1:(num_seg-1)
                audiowrite([output_dir16k '/' data_type{i_type} '/s1/' file_name '_seg_' num2str(seg_i) '.wav'],s1_16k((seg_i-1)*cut_len+1:seg_i*cut_len,1),fs);
                audiowrite([output_dir16k '/' data_type{i_type} '/s2/' file_name '_seg_' num2str(seg_i) '.wav'],s2_16k((seg_i-1)*cut_len+1:seg_i*cut_len,1),fs);
                audiowrite([output_dir16k '/' data_type{i_type} '/mix/' file_name '_seg_' num2str(seg_i) '.wav'],mix_16k((seg_i-1)*cut_len+1:seg_i*cut_len,1),fs);
            end
            if (mix_16k_length - (num_seg-1)*cut_len+1) <= max_len
                audiowrite([output_dir16k '/' data_type{i_type} '/s1/' file_name '_seg_' num2str(num_seg) '.wav'],s1_16k((num_seg-1)*cut_len+1:end,1),fs);
                audiowrite([output_dir16k '/' data_type{i_type} '/s2/' file_name '_seg_' num2str(num_seg) '.wav'],s2_16k((num_seg-1)*cut_len+1:end,1),fs);
                audiowrite([output_dir16k '/' data_type{i_type} '/mix/' file_name '_seg_' num2str(num_seg) '.wav'],mix_16k((num_seg-1)*cut_len+1:end,1),fs);
            else
                audiowrite([output_dir16k '/' data_type{i_type} '/s1/' file_name '_seg_' num2str(num_seg) '.wav'],s1_16k((num_seg-1)*cut_len+1:num_seg*cut_len,1),fs);
                audiowrite([output_dir16k '/' data_type{i_type} '/s2/' file_name '_seg_' num2str(num_seg) '.wav'],s2_16k((num_seg-1)*cut_len+1:num_seg*cut_len,1),fs);
                audiowrite([output_dir16k '/' data_type{i_type} '/mix/' file_name '_seg_' num2str(num_seg) '.wav'],mix_16k((num_seg-1)*cut_len+1:num_seg*cut_len,1),fs);

                audiowrite([output_dir16k '/' data_type{i_type} '/s1/' file_name '_seg_' num2str(num_seg+1) '.wav'],s1_16k(num_seg*cut_len+1:end,1),fs);
                audiowrite([output_dir16k '/' data_type{i_type} '/s2/' file_name '_seg_' num2str(num_seg+1) '.wav'],s2_16k(num_seg*cut_len+1:end,1),fs);
                audiowrite([output_dir16k '/' data_type{i_type} '/mix/' file_name '_seg_' num2str(num_seg+1) '.wav'],mix_16k(num_seg*cut_len+1:end,1),fs);
            end
        end
        
        if mod(file,10)==0
            fprintf(1,'.');
            if mod(file,200)==0
                fprintf(1,'\n');
            end
        end
        
    end
    
end
