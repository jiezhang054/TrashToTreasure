import torch
import torch.nn as nn

from utils.enconder import MultiViewUsefulEncoderSystem,MultiViewTrashEncoderSystem,TreasureRepresentationEncoder,PredictionFusion
from utils.decoder import MultiViewDecoderSystem


class TrashToTreasure(nn.Module):
    def __init__(self, config):
        super(TrashToTreasure, self).__init__()
        self.config = config
        self.mum_views = self.config.num_views

        self.MultiViewUsefulEncoderSystem = MultiViewUsefulEncoderSystem(self.config)
        self.MultiViewTrashEncoderSystem = MultiViewTrashEncoderSystem(self.config)
        self.TreasureRepresentationEncoder = TreasureRepresentationEncoder(self.config)
        self.MultiViewDecoderSystem = MultiViewDecoderSystem(self.config)
        self.PredictionFusion = PredictionFusion(self.config)

    def forward(self, data_list):

        useful_list = self.MultiViewUsefulEncoderSystem(data_list)
        trash_list = self.MultiViewTrashEncoderSystem(data_list)

        treasure = self.TreasureRepresentationEncoder(trash_list)

        T_list = []

        for useful, trash in zip(useful_list, trash_list):
            T_list.append(torch.cat((useful, trash), dim=1))

        reconstructed_list = self.MultiViewDecoderSystem(T_list)

        prediction = self.PredictionFusion(useful_list, treasure)

        res = {
            'original_list': data_list,
            'useful_list': useful_list,
            'trash_list': trash_list,
            'reconstructed_list': reconstructed_list,
            'treasure': treasure,
            'prediction': prediction,
        }
        return res



        











