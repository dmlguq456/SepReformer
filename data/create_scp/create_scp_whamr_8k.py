import os


train_mix_scp = 'scp_ss_8k_whamr/tr_mix.scp'
train_s1_scp = 'scp_ss_8k_whamr/tr_s1.scp'
train_s2_scp = 'scp_ss_8k_whamr/tr_s2.scp'
train_s1_reverb_scp = 'scp_ss_8k_whamr/tr_s1_reverb.scp'
train_s2_reverb_scp = 'scp_ss_8k_whamr/tr_s2_reverb.scp'
train_n_scp = 'scp_ss_8k_whamr/tr_n.scp'


train_mix = '/home/work/data/WHAMR/tr/mix_both_reverb'
train_s1 = '/home/work/data/WHAMR/tr/s1_anechoic'
train_s2 = '/home/work/data/WHAMR/tr/s2_anechoic'
train_s1_reverb = '/home/work/data/WHAMR/tr/s1_reverb'
train_s2_reverb = '/home/work/data/WHAMR/tr/s2_reverb'
train_n = '/home/work/data/WHAMR/tr/noise'


tr_mix = open(train_mix_scp,'w')
for root, dirs, files in os.walk(train_mix):
    files.sort()
    for file in files:
        tr_mix.write(file+" "+root+'/'+file)
        tr_mix.write('\n')


tr_s1 = open(train_s1_scp,'w')
for root, dirs, files in os.walk(train_s1):
    files.sort()
    for file in files:
        tr_s1.write(file+" "+root+'/'+file)
        tr_s1.write('\n')


tr_s2 = open(train_s2_scp,'w')
for root, dirs, files in os.walk(train_s2):
    files.sort()
    for file in files:
        tr_s2.write(file+" "+root+'/'+file)
        tr_s2.write('\n')


tr_s1_reverb = open(train_s1_reverb_scp,'w')
for root, dirs, files in os.walk(train_s1_reverb):
    files.sort()
    for file in files:
        tr_s1_reverb.write(file+" "+root+'/'+file)
        tr_s1_reverb.write('\n')


tr_s2_reverb = open(train_s2_reverb_scp,'w')
for root, dirs, files in os.walk(train_s2_reverb):
    files.sort()
    for file in files:
        tr_s2_reverb.write(file+" "+root+'/'+file)
        tr_s2_reverb.write('\n')



tr_noise = open(train_n_scp,'w')
for root, dirs, files in os.walk(train_n):
    files.sort()
    for file in files:
        tr_noise.write(file+" "+root+'/'+file)
        tr_noise.write('\n')




test_mix_scp = 'scp_ss_8k_whamr/tt_mix.scp'
test_s1_scp = 'scp_ss_8k_whamr/tt_s1.scp'
test_s2_scp = 'scp_ss_8k_whamr/tt_s2.scp'

test_mix = '/home/work/data/WHAMR/tt/mix_both_reverb'
test_s1 = '/home/work/data/WHAMR/tt/s1_anechoic'
test_s2 = '/home/work/data/WHAMR/tt/s2_anechoic'

tt_mix = open(test_mix_scp,'w')
for root, dirs, files in os.walk(test_mix):
    files.sort()
    for file in files:
        tt_mix.write(file+" "+root+'/'+file)
        tt_mix.write('\n')


tt_s1 = open(test_s1_scp,'w')
for root, dirs, files in os.walk(test_s1):
    files.sort()
    for file in files:
        tt_s1.write(file+" "+root+'/'+file)
        tt_s1.write('\n')


tt_s2 = open(test_s2_scp,'w')
for root, dirs, files in os.walk(test_s2):
    files.sort()
    for file in files:
        tt_s2.write(file+" "+root+'/'+file)
        tt_s2.write('\n')

cv_mix_scp = 'scp_ss_8k_whamr/cv_mix.scp'
cv_s1_scp = 'scp_ss_8k_whamr/cv_s1.scp'
cv_s2_scp = 'scp_ss_8k_whamr/cv_s2.scp'

cv_mix = '/home/work/data/WHAMR/cv/mix_both_reverb'
cv_s1 = '/home/work/data/WHAMR/cv/s1_anechoic'
cv_s2 = '/home/work/data/WHAMR/cv/s2_anechoic'

cv_mix_file = open(cv_mix_scp,'w')
for root, dirs, files in os.walk(cv_mix):
    files.sort()
    for file in files:
        cv_mix_file.write(file+" "+root+'/'+file)
        cv_mix_file.write('\n')


cv_s1_file = open(cv_s1_scp,'w')
for root, dirs, files in os.walk(cv_s1):
    files.sort()
    for file in files:
        cv_s1_file.write(file+" "+root+'/'+file)
        cv_s1_file.write('\n')


cv_s2_file = open(cv_s2_scp,'w')
for root, dirs, files in os.walk(cv_s2):
    files.sort()
    for file in files:
        cv_s2_file.write(file+" "+root+'/'+file)
        cv_s2_file.write('\n')
        
        
    