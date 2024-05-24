
data_type = {'dev','test','train-100','train-360'};
data_root = '/home/nas/user/Uihyeop/DB/LibriMix/data/Libri2Mix/wav16k/css'; % YOUR_PATH/, the folder containing data/
output_dir16k='/home/nas/user/Uihyeop/DB/LibriMix/data/Libri2Mix/wav16k/css_segment';

mix_style = 'css_v3_segment_short';

current_dir = cd('.');
for i_type = 1:length(data_type)
    cd([data_root '/' data_type{i_type} '/mix_clean/']); file_list = dir('*.wav'); cd(current_dir);

    if ~exist([output_dir16k '/' data_type{i_type}],'dir')
        mkdir([output_dir16k '/' data_type{i_type}]);
    end
    status = mkdir([output_dir16k '/' data_type{i_type} '/s1/']); %#ok<NASGU>
    status = mkdir([output_dir16k '/' data_type{i_type} '/s2/']); %#ok<NASGU>
    status = mkdir([output_dir16k '/' data_type{i_type} '/mix_clean/']);

    fprintf(1,'%s\n',[mix_style '_' data_type{i_type}]);
    for file = 1:size(file_list,1)
        file_name = file_list(file).name(1:end-4);
        [s1_16k, fs] = audioread([data_root, '/', data_type{i_type},'/s1/', file_name, '.wav']);
        s2_16k       = audioread([data_root, '/', data_type{i_type},'/s2/', file_name,'.wav']);
        mix_16k       = audioread([data_root, '/', data_type{i_type},'/mix_clean/', file_name, '.wav']);

        s1_16k = int16(round((2^15)*s1_16k));
        s2_16k = int16(round((2^15)*s2_16k));
        mix_16k = int16(round((2^15)*mix_16k));
        mix_16k_length = length(mix_16k);
        cut_len = 4*fs;
        max_len = 6*fs;
        if length(mix_16k) <= max_len
            audiowrite([output_dir16k '/' data_type{i_type} '/s1/' file_name '.wav'],s1_16k,fs);
            audiowrite([output_dir16k '/' data_type{i_type} '/s2/' file_name '.wav'],s2_16k,fs);
            audiowrite([output_dir16k '/' data_type{i_type} '/mix_clean/' file_name '.wav'],mix_16k,fs);
        else
            num_seg = fix(length(mix_16k)/(cut_len));
            for seg_i = 1:(num_seg-1)
                audiowrite([output_dir16k '/' data_type{i_type} '/s1/' file_name '_seg_' num2str(seg_i) '.wav'],s1_16k((seg_i-1)*cut_len+1:seg_i*cut_len,1),fs);
                audiowrite([output_dir16k '/' data_type{i_type} '/s2/' file_name '_seg_' num2str(seg_i) '.wav'],s2_16k((seg_i-1)*cut_len+1:seg_i*cut_len,1),fs);
                audiowrite([output_dir16k '/' data_type{i_type} '/mix_clean/' file_name '_seg_' num2str(seg_i) '.wav'],mix_16k((seg_i-1)*cut_len+1:seg_i*cut_len,1),fs);
            end
            if (mix_16k_length - (num_seg-1)*cut_len+1) <= max_len
                audiowrite([output_dir16k '/' data_type{i_type} '/s1/' file_name '_seg_' num2str(num_seg) '.wav'],s1_16k((num_seg-1)*cut_len+1:end,1),fs);
                audiowrite([output_dir16k '/' data_type{i_type} '/s2/' file_name '_seg_' num2str(num_seg) '.wav'],s2_16k((num_seg-1)*cut_len+1:end,1),fs);
                audiowrite([output_dir16k '/' data_type{i_type} '/mix_clean/' file_name '_seg_' num2str(num_seg) '.wav'],mix_16k((num_seg-1)*cut_len+1:end,1),fs);
            else
                audiowrite([output_dir16k '/' data_type{i_type} '/s1/' file_name '_seg_' num2str(num_seg) '.wav'],s1_16k((num_seg-1)*cut_len+1:num_seg*cut_len,1),fs);
                audiowrite([output_dir16k '/' data_type{i_type} '/s2/' file_name '_seg_' num2str(num_seg) '.wav'],s2_16k((num_seg-1)*cut_len+1:num_seg*cut_len,1),fs);
                audiowrite([output_dir16k '/' data_type{i_type} '/mix_clean/' file_name '_seg_' num2str(num_seg) '.wav'],mix_16k((num_seg-1)*cut_len+1:num_seg*cut_len,1),fs);

                audiowrite([output_dir16k '/' data_type{i_type} '/s1/' file_name '_seg_' num2str(num_seg+1) '.wav'],s1_16k(num_seg*cut_len+1:end,1),fs);
                audiowrite([output_dir16k '/' data_type{i_type} '/s2/' file_name '_seg_' num2str(num_seg+1) '.wav'],s2_16k(num_seg*cut_len+1:end,1),fs);
                audiowrite([output_dir16k '/' data_type{i_type} '/mix_clean/' file_name '_seg_' num2str(num_seg+1) '.wav'],mix_16k(num_seg*cut_len+1:end,1),fs);
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
