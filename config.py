import os, math

class Config(object):
    '''Main function'''
    # Learning
    lr = 2.5e-4
    decay = math.pow(0.5, 1/20)
    epochs = 200
    patience = 15
    save_dir = "snapshot"
    batch_size = 1
    weight_decay = 0

    # Data
    dataset = 'IEMOCAP'                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
    wr = 0.05 # weight_rate
    key = "emotion"# ["emotion", "intent", "ee", "er", "emo", "act"]
    loss_mask = "data/"

    # model
    multi = False
    fusionModule = "GF"
    pre_fusion_dropout = 0.5
    post_fusion_dropout = 0.5
    fusion_input_features = [300, 512, 100]
    fusion_utterance_dim = None
    pre_fusion_hidden_dims = [30, 30, 30]



    roberta = False
    embedding_train = False
    d_word_vec = 300
    d_h1 = 300
    d_h2 = 300
    drop = 0.5
    ll = 2 # gat layer
    gpu = "1"
    seed = 1234
    report_lr = 1000
    num_workers = 0
    num_classes = None

    #path
    if multi:
        data_root = "Data/"
    else:
        data_root = "data/"
    
    vocab_root = "data/vocabs/"
    embedding_root = "data/embeddings/"