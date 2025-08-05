# a2fseg_1.Generic_MAML_multi3_channel(4, 32, 3, 5, 2, 2, torch.nn.modules.conv.Conv3d, torch.nn.modules.instancenorm.InstanceNorm3d, {'eps': '1e-05', 'affine': True}, torch.nn.modules.dropout.Dropout3d, {'p': 0, 'inplace': True}, torch.nn.modules.activation.LeakyReLU, {'negative_slope': 0.01, 'inplace': True}, True, False, lambda x: x, InitWeights_He(1e-2), [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]], [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]], False, True, True)

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.nets.a2fseg_1 import Generic_MAML_multi3_channel
from torch import nn
from torch.optim import Adam, AdamW
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
import torch
from nnunetv2.nets.a2fseg.initialization import InitWeights_He

class nnUNetTrainerA2Fseg(nnUNetTrainer):
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

        # model = Generic_MAML_multi3_channel(
        #     modality_num = num_input_channels,
        #     base_num_features = 32,
        #     num_classes = len(dataset_json['labels']),
        #     num_pool = 5,
        #     num_conv_per_stage = 2,
        #     feat_map_mul_on_downscale = 2,
        #     conv_op = conv_op,
        #     norm_op = norm_op,
        #     norm_op_kwargs = norm_op_kwargs,
        #     dropout_op = dropout_op,
        #     dropout_op_kwargs = dropout_op_kwargs,
        #     nonlin = net_nonlin,
        #     nonlin_kwargs = net_nonlin_kwargs,
        #     deep_supervision = True,
        #     dropout_in_localization = False,
        #     final_nonlin = lambda x: x,
        #     weightInitializer = InitWeights_He(1e-2),
        #     pool_op_kernel_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        #     conv_kernel_sizes = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        #     upscale_logits = False, convolutional_pooling = True, convolutional_upsampling = True
        # )
        model = Generic_MAML_multi3_channel(
            modality_num = num_input_channels,
            base_num_features = 32,
            num_classes = len(dataset_json['labels']),
            num_pool = len(configuration_manager.pool_op_kernel_sizes) - 1,
            num_conv_per_stage = 2,
            feat_map_mul_on_downscale = 2,
            conv_op = conv_op,
            norm_op = norm_op,
            norm_op_kwargs = norm_op_kwargs,
            dropout_op = dropout_op,
            dropout_op_kwargs = dropout_op_kwargs,
            nonlin = net_nonlin,
            nonlin_kwargs = net_nonlin_kwargs,
            deep_supervision = True,
            dropout_in_localization = False,
            final_nonlin = lambda x: x,
            weightInitializer = InitWeights_He(1e-2),
            pool_op_kernel_sizes = configuration_manager.pool_op_kernel_sizes[1:],
            conv_kernel_sizes = configuration_manager.conv_kernel_sizes,
            upscale_logits = False, convolutional_pooling = True, convolutional_upsampling = True
        )

        return model
    
    def configure_optimizers(self):
        optimizer = AdamW(self.network.parameters(),
                          lr = self.initial_lr,
                          weight_decay = self.weight_decay,
                          amsgrad = True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)

        return optimizer, lr_scheduler