import torch
import os
import inspect
import gentp

class Configs():
    def __init__(self):
        # Library configs
        self.base_path = '/media/jeremy/cde0dfff-70f1-4c1c-82aa-e0d469c14c62/GenTP_submission_backup/igbsonreplay/release/GenTP-dev/gentp'
        self.igibson_path = '/media/jeremy/cde0dfff-70f1-4c1c-82aa-e0d469c14c62/GenTP_submission_backup/igbsonreplay/release/iGibson'

        # dataset configs
        self.pretrain_train_data_path = '{}/datasets/data_pretrain_train_10000.hdf5'.format(self.base_path)
        self.pretrain_test_data_path = '{}/datasets/data_pretrain_test_1000.hdf5'.format(self.base_path)
        self.trans_train_data_path = '{}/datasets/data_trans_train_1000.hdf5'.format(self.base_path)
        self.trans_test_data_path = '{}/datasets/data_trans_test_100.hdf5'.format(self.base_path)

        # training configs
        self.batch_size = 8
        self.eval_batch_size = 1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # model configs
        self.emsize = 256  # embedding dimension
        self.d_hid = 256  # dimension of the feedforward network model in nn.TransformerEncoder
        self.nlayers = 4  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        self.nhead = 2  # number of heads in nn.MultiheadAttention
        self.dropout = 0.01  # dropout probability
        self.lr = 0.0001  # learning rate
        self.epochs = 500

        # environment configs
        self.env_config_file = "behavior_full_observability_fetch_cooking.yaml"
        self.buffer_size = 120
        self.num_item = 3
        self.num_candidate = 2
        self.bfs_depth = 1
        self.object_num = 7
        self.num_act = 6
        self.num_future_act = 2
        self.num_syb = 14
        self.data_collect_max_step = 12
        self.image_size = 128
        self.obj_id_list = [244, 245, 246, 247, 248, 81, 82]

        # data generation environment configs
        self.env_datagen_config_file = "behavior_full_observability_fetch_datacollection_cooking.yaml"
        self.neglect_obj = ['floors', 'pot_plant_29', 'straight_chair_20', 'straight_chair_21', 'straight_chair_22', 'straight_chair_23', 'walls', 'window_50', 'shelf_10', 'countertop_56']
        self.surface_list = ['countertop_76', 'burner_77', 'frying_pan']
        self.datagen_num_frames = 500 # for pretrain dataset
        self.datagen_vis = False
        self.datagen_trans_num_epochs = 1000 # for transition model dataset
        self.datagen_bfs_depth = 1
        self.datagen_trans_max_steps = 12

        # Pretrained representation (only used during training the transition model)
        self.pretrained_model = '{}/trained_models/pre_best_model_cookable_layout_1_0.1552749971548716.pth'.format(self.base_path)

        # Trained transition model (only used during evaluation)
        self.eval_model_list = ['{}/released_trained_models/cooking_ours_pre_2_113_0.6435430292729978.pth'.format(self.base_path), ]