import numpy as np
import re
import itertools
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


#def load_data_and_labels():
#    """
#    Loads MR polarity data from files, splits the data into words and generates labels.
#    Returns split sentences and labels.
#    """
#    # Load data from files
#    positive_examples = list(open("./data/rt-polaritydata/rt-polarity.pos", "r").readlines())
#    positive_examples = [s.strip() for s in positive_examples]
#    negative_examples = list(open("./data/rt-polaritydata/rt-polarity.neg", "r").readlines())
#    negative_examples = [s.strip() for s in negative_examples]
#    # Split by words
#    x_text = positive_examples + negative_examples
#    x_text = [clean_str(sent) for sent in x_text]
##   # Generate labels
#    positive_labels = [[0, 1] for _ in positive_examples]
#    negative_labels = [[1, 0] for _ in negative_examples]
##    y = np.concatenate([positive_labels, negative_labels], 0)
##    return [x_text, y]


def load_data_and_labels():

    talk_religion_misc_data = list(open("./clean/train_clean/talk.religion.misc.txt","r").readlines())
    talk_religion_misc_data = [s.strip() for s in talk_religion_misc_data]
    alt_atheism_data = list(open("./clean/train_clean/alt.atheism.txt","r").readlines())
    alt_atheism_data = [s.strip() for s in alt_atheism_data]
    comp_graphics_data = list(open("./clean/train_clean/comp.graphics.txt","r").readlines())
    comp_graphics_data = [s.strip() for s in comp_graphics_data]
    comp_os_ms_windows_misc_data = list(open("./clean/train_clean/comp.os.ms-windows.misc.txt","r").readlines())
    comp_os_ms_windows_misc_data = [s.strip() for s in comp_os_ms_windows_misc_data]
    comp_sys_ibm_pc_hardware_data = list(open("./clean/train_clean/comp.sys.ibm.pc.hardware.txt","r").readlines())
    comp_sys_ibm_pc_hardware_data = [s.strip() for s in comp_sys_ibm_pc_hardware_data]
    comp_sys_mac_hardware_data = list(open("./clean/train_clean/comp.sys.mac.hardware.txt","r").readlines())
    comp_sys_mac_hardware_data = [s.strip() for s in comp_sys_mac_hardware_data]
    comp_windows_x_data = list(open("./clean/train_clean/comp.windows.x.txt","r").readlines())
    comp_windows_x_data = [s.strip() for s in comp_windows_x_data]
    misc_forsale_data = list(open("./clean/train_clean/misc.forsale.txt","r").readlines())
    misc_forsale_data = [s.strip() for s in misc_forsale_data]
    rec_autos_data = list(open("./clean/train_clean/rec.autos.txt","r").readlines())
    rec_autos_data = [s.strip() for s in rec_autos_data]
    rec_motorcycles_data = list(open("./clean/train_clean/rec.motorcycles.txt","r").readlines())
    rec_motorcycles_data = [s.strip() for s in rec_motorcycles_data]
    rec_sport_baseball_data = list(open("./clean/train_clean/rec.sport.baseball.txt","r").readlines())
    rec_sport_baseball_data = [s.strip() for s in rec_sport_baseball_data]
    rec_sport_hockey_data = list(open("./clean/train_clean/rec.sport.hockey.txt","r").readlines())
    rec_sport_hockey_data = [s.strip() for s in rec_sport_hockey_data]
    sci_crypt_data = list(open("./clean/train_clean/sci.crypt.txt","r").readlines())
    sci_crypt_data = [s.strip() for s in sci_crypt_data]
    sci_electronics_data = list(open("./clean/train_clean/sci.electronics.txt","r").readlines())
    sci_electronics_data = [s.strip() for s in sci_electronics_data]
    sci_med_data = list(open("./clean/train_clean/sci.med.txt","r").readlines())
    sci_med_data = [s.strip() for s in sci_med_data]
    sci_space_data = list(open("./clean/train_clean/sci.space.txt","r").readlines())
    sci_space_data = [s.strip() for s in sci_space_data]
    soc_religion_christian_data = list(open("./clean/train_clean/soc.religion.christian.txt","r").readlines())
    soc_religion_christian_data = [s.strip() for s in soc_religion_christian_data]
    talk_politics_guns_data = list(open("./clean/train_clean/talk.politics.guns.txt","r").readlines())
    talk_politics_guns_data = [s.strip() for s in talk_politics_guns_data]
    talk_politics_mideast_data = list(open("./clean/train_clean/talk.politics.mideast.txt","r").readlines())
    talk_politics_mideast_data = [s.strip() for s in talk_politics_mideast_data]
    talk_politics_misc_data = list(open("./clean/train_clean/talk.politics.misc.txt","r").readlines())
    talk_politics_misc_data = [s.strip() for s in talk_politics_misc_data]

    talk_religion_misc_labels =         [[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] for _ in talk_religion_misc_data]
    alt_atheism_labels =                [[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] for _ in alt_atheism_data]
    comp_graphics_labels =              [[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] for _ in comp_graphics_data]
    comp_os_ms_windows_misc_labels  =   [[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] for _ in comp_os_ms_windows_misc_data]
    comp_sys_ibm_pc_hardware_labels =   [[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] for _ in comp_sys_ibm_pc_hardware_data]
    comp_sys_mac_hardware_labels =      [[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0] for _ in comp_sys_mac_hardware_data]
    comp_windows_x_labels =             [[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0] for _ in comp_windows_x_data]
    misc_forsale_labels =               [[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0] for _ in misc_forsale_data]
    rec_autos_labels =                  [[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0] for _ in rec_autos_data]
    rec_motorcycles_labels =            [[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0] for _ in rec_motorcycles_data]
    rec_sport_baseball_labels =         [[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0] for _ in rec_sport_baseball_data]
    rec_sport_hockey_labels =           [[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0] for _ in rec_sport_hockey_data]
    sci_crypt_labels =                  [[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0] for _ in sci_crypt_data]
    sci_electronics_labels =            [[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0] for _ in sci_electronics_data]
    sci_med_labels =                    [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0] for _ in sci_med_data]
    sci_space_labels =                  [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0] for _ in sci_space_data]
    soc_religion_christian_labels =     [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0] for _ in soc_religion_christian_data]
    talk_politics_guns_labels =         [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0] for _ in talk_politics_guns_data]
    talk_politics_mideast_labels =      [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0] for _ in talk_politics_mideast_data]
    talk_politics_misc_labels =         [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1] for _ in talk_politics_misc_data]        



    labels = np.concatenate([talk_religion_misc_labels,alt_atheism_labels ,
    comp_graphics_labels ,
    comp_os_ms_windows_misc_labels,
    comp_sys_ibm_pc_hardware_labels,
    comp_sys_mac_hardware_labels ,
    comp_windows_x_labels ,
    misc_forsale_labels ,
    rec_autos_labels,
    rec_motorcycles_labels,
    rec_sport_baseball_labels,
    rec_sport_hockey_labels,
    sci_crypt_labels,
    sci_electronics_labels,
    sci_med_labels,
    sci_space_labels,
    soc_religion_christian_labels,
    talk_politics_guns_labels,
    talk_politics_mideast_labels,
    talk_politics_misc_labels,],0)

    labels.shape

    X =  np.concatenate(
    [talk_religion_misc_data,
    alt_atheism_data ,
    comp_graphics_data ,
    comp_os_ms_windows_misc_data,
    comp_sys_ibm_pc_hardware_data,
    comp_sys_mac_hardware_data ,
    comp_windows_x_data ,
    misc_forsale_data ,
    rec_autos_data,
    rec_motorcycles_data,
    rec_sport_baseball_data,
    rec_sport_hockey_data,
    sci_crypt_data,
    sci_electronics_data,
    sci_med_data,
    sci_space_data,
    soc_religion_christian_data,
    talk_politics_guns_data,
    talk_politics_mideast_data,
    talk_politics_misc_data],0)
    return ([X,labels])
#######################
def load_test_data_and_labels():

    talk_religion_misc_data = list(open("./clean/test_clean/alt.atheism.txt","r").readlines())
    talk_religion_misc_data = [s.strip() for s in talk_religion_misc_data]
    alt_atheism_data = list(open("./clean/test_clean/comp.graphics.txt","r").readlines())
    alt_atheism_data = [s.strip() for s in alt_atheism_data]
    comp_graphics_data = list(open("./clean/test_clean/comp.os.ms-windows.misc.txt","r").readlines())
    comp_graphics_data = [s.strip() for s in comp_graphics_data]
    comp_os_ms_windows_misc_data = list(open("./clean/test_clean/comp.sys.ibm.pc.hardware.txt","r").readlines())
    comp_os_ms_windows_misc_data = [s.strip() for s in comp_os_ms_windows_misc_data]
    comp_sys_ibm_pc_hardware_data = list(open("./clean/test_clean/comp.sys.mac.hardware.txt","r").readlines())
    comp_sys_ibm_pc_hardware_data = [s.strip() for s in comp_sys_ibm_pc_hardware_data]
    comp_sys_mac_hardware_data = list(open("./clean/test_clean/comp.windows.x.txt","r").readlines())
    comp_sys_mac_hardware_data = [s.strip() for s in comp_sys_mac_hardware_data]
    comp_windows_x_data = list(open("./clean/test_clean/misc.forsale.txt","r").readlines())
    comp_windows_x_data = [s.strip() for s in comp_windows_x_data]
    misc_forsale_data = list(open("./clean/test_clean/rec.autos.txt","r").readlines())
    misc_forsale_data = [s.strip() for s in misc_forsale_data]
    rec_autos_data = list(open("./clean/test_clean/rec.motorcycles.txt","r").readlines())
    rec_autos_data = [s.strip() for s in rec_autos_data]
    rec_motorcycles_data = list(open("./clean/test_clean/rec.sport.baseball.txt","r").readlines())
    rec_motorcycles_data = [s.strip() for s in rec_motorcycles_data]
    rec_sport_baseball_data = list(open("./clean/test_clean/rec.sport.hockey.txt","r").readlines())
    rec_sport_baseball_data = [s.strip() for s in rec_sport_baseball_data]
    rec_sport_hockey_data = list(open("./clean/test_clean/sci.crypt.txt","r").readlines())
    rec_sport_hockey_data = [s.strip() for s in rec_sport_hockey_data]
    sci_crypt_data = list(open("./clean/test_clean/sci.electronics.txt","r").readlines())
    sci_crypt_data = [s.strip() for s in sci_crypt_data]
    sci_electronics_data = list(open("./clean/test_clean/sci.med.txt","r").readlines())
    sci_electronics_data = [s.strip() for s in sci_electronics_data]
    sci_med_data = list(open("./clean/test_clean/sci.space.txt","r").readlines())
    sci_med_data = [s.strip() for s in sci_med_data]
    sci_space_data = list(open("./clean/test_clean/soc.religion.christian.txt","r").readlines())
    sci_space_data = [s.strip() for s in sci_space_data]
    soc_religion_christian_data = list(open("./clean/test_clean/talk.politics.guns.txt","r").readlines())
    soc_religion_christian_data = [s.strip() for s in soc_religion_christian_data]
    talk_politics_guns_data = list(open("./clean/test_clean/talk.politics.mideast.txt","r").readlines())
    talk_politics_guns_data = [s.strip() for s in talk_politics_guns_data]
    talk_politics_mideast_data = list(open("./clean/test_clean/talk.politics.misc.txt","r").readlines())
    talk_politics_mideast_data = [s.strip() for s in talk_politics_mideast_data]
    talk_politics_misc_data = list(open("./clean/test_clean/talk.religion.misc.txt","r").readlines())
    talk_politics_misc_data = [s.strip() for s in talk_politics_misc_data]

    talk_religion_misc_labels = [[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] for _ in talk_religion_misc_data]
    alt_atheism_labels =        [[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] for _ in alt_atheism_data]
    comp_graphics_labels =      [[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] for _ in comp_graphics_data]
    comp_os_ms_windows_misc_labels  = [[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] for _ in comp_os_ms_windows_misc_data]
    comp_sys_ibm_pc_hardware_labels = [[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] for _ in comp_sys_ibm_pc_hardware_data]
    comp_sys_mac_hardware_labels =[[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0, 0,0,0,0,0] for _ in comp_sys_mac_hardware_data]
    comp_windows_x_labels =     [[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0] for _ in comp_windows_x_data]
    misc_forsale_labels =       [[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0] for _ in misc_forsale_data]
    rec_autos_labels =          [[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0] for _ in rec_autos_data]
    rec_motorcycles_labels =    [[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0] for _ in rec_motorcycles_data]
    rec_sport_baseball_labels = [[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0] for _ in rec_sport_baseball_data]
    rec_sport_hockey_labels =   [[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0] for _ in rec_sport_hockey_data]
    sci_crypt_labels =          [[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0] for _ in sci_crypt_data]
    sci_electronics_labels =    [[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0] for _ in sci_electronics_data]
    sci_med_labels =            [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0] for _ in sci_med_data]
    sci_space_labels =          [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0] for _ in sci_space_data]
    soc_religion_christian_labels = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,1,0,0,0] for _ in soc_religion_christian_data]
    talk_politics_guns_labels = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0] for _ in talk_politics_guns_data]
    talk_politics_mideast_labels = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,1,0] for _ in talk_politics_mideast_data]
    talk_politics_misc_labels = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1] for _ in talk_politics_misc_data]



    labels = np.concatenate([talk_religion_misc_labels,alt_atheism_labels ,
    comp_graphics_labels ,
    comp_os_ms_windows_misc_labels,
    comp_sys_ibm_pc_hardware_labels,
    comp_sys_mac_hardware_labels ,
    comp_windows_x_labels ,
    misc_forsale_labels ,
    rec_autos_labels,
    rec_motorcycles_labels,
    rec_sport_baseball_labels,
    rec_sport_hockey_labels,
    sci_crypt_labels,
    sci_electronics_labels,
    sci_med_labels,
    sci_space_labels,
    soc_religion_christian_labels,
    talk_politics_guns_labels,
    talk_politics_mideast_labels,
    talk_politics_misc_labels,],0)

    labels.shape

    X =  np.concatenate(
    [talk_religion_misc_data,
    alt_atheism_data ,
    comp_graphics_data ,
    comp_os_ms_windows_misc_data,
    comp_sys_ibm_pc_hardware_data,
    comp_sys_mac_hardware_data ,
    comp_windows_x_data ,
    misc_forsale_data ,
    rec_autos_data,
    rec_motorcycles_data,
    rec_sport_baseball_data,
    rec_sport_hockey_data,
    sci_crypt_data,
    sci_electronics_data,
    sci_med_data,
    sci_space_data,
    soc_religion_christian_data,
    talk_politics_guns_data,
    talk_politics_mideast_data,
    talk_politics_misc_data],0)
    return ([X,labels])
#######################
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
