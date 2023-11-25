import os
import pickle
import copy
import numpy as np
from typing import Callable, Optional
import torch
import sys
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer

sbert_model = SentenceTransformer("paraphrase-distilroberta-base-v1")


def read_data():
    data_file = "/mnt/disk1/hangpt/Multimodal-FL/FLMultimodal-PTH/benchmark/RAW_DATA/IEMOCAP_COGMEN/IEMOCAP_features.pkl"
    video_ids, video_speakers, video_labels, video_text, \
        video_audio, video_visual, video_sentence, \
        trainVids, testVids = pickle.load(open(data_file,"rb"), encoding='latin1')
    # utterance eg: 'Ses02F_script01_3'
    video_sentence_embeds = copy.deepcopy(video_sentence)
    # texts = copy.deepcopy(video_text)
    num_utterances = [len(video_audio[vid]) for vid in trainVids]
    total_num_utterances = sum(num_utterances)
    print(np.unique(num_utterances), total_num_utterances)

    audios = copy.deepcopy(video_audio)
    visuals = copy.deepcopy(video_visual)

    count_ses = [0]*4
    for vid in trainVids:
        if vid.startswith('Ses01'):
            count_ses[0] += 1
        elif vid.startswith('Ses02'):
            count_ses[1] += 1
        elif vid.startswith('Ses03'):
            count_ses[2] += 1
        else:
            count_ses[3] += 1
    print(count_ses, sum(count_ses))
    # import pdb; pdb.set_trace()
    # train_data = torch.zeros((total_num_utterances, 3, 768))
    train_data = []
    train_label = []
    for vid in trainVids:
        num_utterances = len(video_audio[vid])
        # import pdb; pdb.set_trace()
        for i in range(num_utterances):
            data = torch.zeros((3,768))
            video_sentence_embeds = torch.tensor(sbert_model.encode(video_sentence[vid][i]))    # (#utterances - different for each vid, 768)
            data[0,:] = video_sentence_embeds   #text
            audios = torch.tensor(video_audio[vid][i])                                            # #utterances, 100
            data[1,:100] = audios
            visuals = torch.tensor(video_visual[vid][i])                                      # #utterances, 512
            data[2,:512] = visuals
            # print(video_sentence_embeds.shape, audios.shape, visuals.shape)      # (768, 100, 512) - 1380
            train_data.append(data)
            train_label.append(video_labels[vid][i])
        
        # import pdb; pdb.set_trace()

    train_data = torch.stack(train_data)
    train_label = torch.tensor(train_label)
    torch.save(train_data, '/mnt/disk1/hangpt/Multimodal-FL/FLMultimodal-PTH/benchmark/RAW_DATA/IEMOCAP_COGMEN/x_train.pt')
    torch.save(train_label, '/mnt/disk1/hangpt/Multimodal-FL/FLMultimodal-PTH/benchmark/RAW_DATA/IEMOCAP_COGMEN/y_train.pt')



    # for vid in trainVids:
    #     # texts[vid] = torch.tensor(video_text[vid])
    #     # import pdb; pdb.set_trace()
    #     # data = torch.zeros((3,768))
    #     video_sentence_embeds[vid] = torch.tensor(sbert_model.encode(video_sentence[vid]))    # (#utterances - different for each vid, 768)
    #     # data[0,:] = video_sentence_embeds
    #     audios[vid] = torch.tensor(video_audio[vid])                                            # #utterances, 100
    #     # data[1,:100] = audios
    #     visuals[vid] = torch.tensor(video_visual[vid])                                      # #utterances, 512
    #     # data[2,:512] = visuals
    #     print(video_sentence_embeds[vid].shape, audios[vid].shape, visuals[vid].shape)      # (768, 100, 512) - 1380
    #     # train_data.append(data)
    #     import pdb; pdb.set_trace()

    test_data = []
    test_label = []
    for vid in testVids:
        num_utterances = len(video_audio[vid])
        # import pdb; pdb.set_trace()
        for i in range(num_utterances):
            data = torch.zeros((3,768))
            video_sentence_embeds = torch.tensor(sbert_model.encode(video_sentence[vid][i]))    # (#utterances - different for each vid, 768)
            data[0,:] = video_sentence_embeds   #text
            audios = torch.tensor(video_audio[vid][i])                                            # #utterances, 100
            data[1,:100] = audios
            visuals = torch.tensor(video_visual[vid][i])                                      # #utterances, 512
            data[2,:512] = visuals
            # print(video_sentence_embeds.shape, audios.shape, visuals.shape)      # (768, 100, 512) - 1380
            test_data.append(data)
            test_label.append(video_labels[vid][i])
            # import pdb; pdb.set_trace()

    test_data = torch.stack(test_data)
    test_label = torch.tensor(test_label)
    torch.save(test_data, '/mnt/disk1/hangpt/Multimodal-FL/FLMultimodal-PTH/benchmark/RAW_DATA/IEMOCAP_COGMEN/x_test.pt')
    torch.save(test_label, '/mnt/disk1/hangpt/Multimodal-FL/FLMultimodal-PTH/benchmark/RAW_DATA/IEMOCAP_COGMEN/y_test.pt')


def gen_list_modalities (missing_rate=0.5, num_modalities=3, NUM_USER=20):
    mat_modals = []
    list_modals_tuples = []
    # num_sample_modals = np.zeros(num_modalities)
    for i in range(NUM_USER):
        unimodal_ind = np.random.randint(num_modalities, size=1)
        # print("Uni: ", np.random.randint(num_modalities, size=1))
        modal_list = np.random.binomial(size=num_modalities, n=1, p=1-missing_rate)
        # modal_indexes = np.where(modal_list==1)[0]
        modal_list[unimodal_ind] = 1
        modal_indexes = np.where(modal_list==1)[0]
        # print(modal_indexes)
        list_modals_tuples.append(tuple(modal_indexes))  
        mat_modals.append(modal_list.tolist())
    mat_modals = np.array(mat_modals)
    num_sample_modals = np.sum(mat_modals, axis=0)
    print("Num_sam:", num_sample_modals)
    print(list_modals_tuples)
    return list_modals_tuples      
    
if __name__ == '__main__':
    # read_data()
    gen_list_modalities(0.5, 3, 20)