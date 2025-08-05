# patch_size (128, 128, 128) or (16, 128, 128, 128)

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.nets.hdenseformer.hdenseformer import HDenseFormer
from torch import nn
from torch.optim import Adam, AdamW
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
import torch

class nnUNetTrainerHDenseFormer(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),    
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.enable_deep_supervision = True
        self.initial_lr = 1e-4
        self.weight_decay = 1e-5
        self.num_epochs = 300

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well!
        """
        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network

    @staticmethod
    def build_network_architecture(plans_manager,
                                   dataset_json,
                                   configuration_manager,
                                   num_input_channels,
                                   enable_deep_supervision):
        if len(configuration_manager.patch_size) == 3:
            conv_op = nn.Conv3d
            norm_op = nn.InstanceNorm3d
            dropout_op = nn.Dropout3d
        else:
            conv_op = nn.Conv2d
            norm_op = nn.InstanceNorm2d
            dropout_op = nn.Dropout2d
        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.modules.activation.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 0.01, 'inplace': True}

        model = HDenseFormer(in_channels = num_input_channels, n_cls = len(dataset_json['labels']), \
                             n_filters = 32, image_size = configuration_manager.patch_size, transformer_depth = 12)

        return model
    
    def configure_optimizers(self):
        optimizer = AdamW(self.network.parameters(),
                          lr = self.initial_lr,
                          weight_decay = self.weight_decay,
                          amsgrad = True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)

        return optimizer, lr_scheduler