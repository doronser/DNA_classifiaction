import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# import pandas as pd
# import cv2

####################################
###        Visualisations        ###
####################################
def seq_imshow(seq_img, label, id):
    """
    Visualise elements from the dataset
    seq_img is a one-hot encoded image
    label 0=bacteria, 1=phage
    id = ID of the DNA sequnece
    """

    plt.figure(figsize=(10, 10))
    plt.imshow(seq_img, interpolation='nearest', aspect='auto',cmap=plt.get_cmap("copper"))
    label_dict = {0: "Bacteria", 1:"Phage"}
    plt.title(f"ID#{id}: {label_dict[label.numpy()]}")



def seq_imshow_crop(seq_img, label, id):
    """
    Visualise elements from the dataset without the padding
    """

    last_contig = np.max(np.nonzero(seq_img.numpy()))
    seq_img_cropped = seq_img[:last_contig,]
    seq_imshow(seq_img_cropped, label, id)


####################################
###        Data Handling         ###
####################################
def get_training_labels(file_path, labels_length, print_every=100):
    """
    extract labels from fasta file sequence names
    """
    labels = np.zeros((labels_length,),dtype=int)
    ids = np.zeros((labels_length,),dtype=int)
    labels_dict = {"bacteria": 0, "phage": 1}
    i = 0
    with open(file_path,"r") as fasta_file:
        fasta_lines = fasta_file.read().split("\n")
        for line in fasta_lines:
            if line.startswith(">"):
                label, id = line.split("-")
                label = label.strip(">").lower()
                labels[i] = labels_dict[label]
                ids[i] = id
                i = i + 1
                if i % print_every == 0:
                    print(f"parsed {i} contigs")
    return labels, ids

def split_seq(seq, label,DNA_SEQUENCE_CLIP=500):
    """
    Split input sequence into sub-sequences of length DNA_SEQUENCE_CLIP
    and duplitcate the label for each sub-sequence.
    Number of splits is the largest whole number of splits possile (remainder ignored).
        seq - one-hot encoded, zero-padded DNA sequence
        label - integer label for the input sequence
        DNA_SEQUENCE_CLIP - length of output sub-sequences
    """
    last_contig = tf.reduce_max(tf.experimental.numpy.nonzero(seq)).numpy()
    num_splits = last_contig//DNA_SEQUENCE_CLIP
    # duplicate the label to match new number of sequences
    dup_label = [label]*num_splits

    # split DNA sequence to maximum number of splits possible without zero padding
    split_seq_list = []
    for i in range(num_splits):
        start_idx = i*DNA_SEQUENCE_CLIP
        end_idx = (i+1)*DNA_SEQUENCE_CLIP
        split_seq_list.append(seq[start_idx:end_idx,:])
    return np.array(split_seq_list), dup_label



def create_dataset(seqs,labels):
    """
    Creates a dataset from pairs of DNA sequences and labels.
    Each sequence is split into a fixed length using split_seq.
    Returns a tf.data.Dataset object
        seqs -  one-hot encoded, zero-padded DNA sequences.
        labels - integer labels of each sequence.
    """
    seqs_list= []
    labels_list = []
    for i in range(seqs.shape[0]):
        new_seq, new_label = split_seq(seqs[i,:,:],labels[i])
        seqs_list.append(new_seq)
        labels_list.append(new_label)
    seqs_split = np.concatenate(seqs_list,axis=0)
    labels_split = np.concatenate(labels_list,axis=0)
    dataset_split = tf.data.Dataset.from_tensor_slices((seqs_split, labels_split))
    print(f"building dataset from {seqs.shape[0]} sequences split into {seqs_split.shape[0]} sequences of length 500")
    return dataset_split


####################################
###           Inference          ###
####################################
def get_test_labels(file_path, labels_length, print_every=100):
    """
    extract ids from test fasta file sequence names
    """
    ids = np.zeros((labels_length,),dtype=int)
    i = 0
    with open(file_path,"r") as fasta_file:
        fasta_lines = fasta_file.read().split("\n")
        for line in fasta_lines:
            if line.startswith(">"):
                id = int(line.strip(">"))
                ids[i] = id
                i = i + 1
                if i % print_every == 0:
                    print(f"parsed {i} contigs")
    return ids



def get_preds(seqs, model, DNA_SEQUENCE_CLIP=500):
    """
    generate predictions for test set
    for each sequence:
        - split to fixed length sequences
        - generate prediction for each slice
        - final prediction is the mean prediction of all slices
    """
    num_seqs = seqs.shape[0]
    final_preds = np.zeros((num_seqs,))
    
    # loop over all sequences
    for seq_num in range(num_seqs):
        if seq_num%100==0:
            print(f"parsed {seq_num} sequences")
        curr_length = np.max(np.nonzero(seqs[seq_num]))
        curr_num_slices = curr_length//DNA_SEQUENCE_CLIP
        # print(f"sequence ID#{test_ids[i]} has length {curr_length} and will be split into {curr_num_slices} slices")
        seq_preds = np.zeros((curr_num_slices,))
        for i in range(curr_num_slices):
        #loop over all slices
            # generate curr slice
            start_idx = i*DNA_SEQUENCE_CLIP
            end_idx = (i+1)*DNA_SEQUENCE_CLIP
            curr_slice = np.expand_dims(seqs[seq_num,start_idx:end_idx,:],axis=0)
            # generate prediction for current slice
            curr_pred = model.predict(curr_slice)
            seq_preds[i] = curr_pred
        
        # final prediction for each sequence is mean of all it's sliced predictions
        final_preds[seq_num]= seq_preds.mean()
    return final_preds

def pred2class(pred):
    if pred>0.5:
        return "Phage"
    else:
        return "Bacteria"