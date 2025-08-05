from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.nets.kan import UKAN
from torch import nn
from torch.optim import Adam, AdamW
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
import torch

class nnUNetTrainerUKAN(nnUNetTrainer):
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
        self.enable_deep_supervision = False
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
        model = UKAN(
            num_classes = len(dataset_json['labels']),
            input_channels=num_input_channels, 
            deep_supervision=False, 
            img_size=configuration_manager.patch_size[0], 
            patch_size=16, 
            in_chans=3, 
            embed_dims=[256, 320, 512], 
            no_kan=False,
            drop_rate=0., 
            drop_path_rate=0., 
            norm_layer=nn.LayerNorm, 
            depths=[1, 1, 1]
        )
        return model
    
    def configure_optimizers(self):
        optimizer = AdamW(self.network.parameters(),
                          lr=self.initial_lr,
                          weight_decay=self.weight_decay,
                          amsgrad=True)

        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler   
