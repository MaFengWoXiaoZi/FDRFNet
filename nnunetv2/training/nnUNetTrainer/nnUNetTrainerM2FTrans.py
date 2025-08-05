# patch_size (128, 128, 128) or (16, 256, 256)

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.nets.m2ftrans import m2ftrans_brats, m2ftrans_prostate, m2ftrans_mmwhs
from torch import nn
from torch.optim import Adam, AdamW
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
import torch
from nnunetv2.nets import criterions

class nnUNetTrainerM2FTrans(nnUNetTrainer):
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
        if dataset_json['name'] == 'BRATS':
            model = m2ftrans_brats.Model(num_cls = len(dataset_json['labels']))
        elif dataset_json['name'] == 'PROSTATE':
            model = m2ftrans_prostate.Model(num_cls = len(dataset_json['labels']))
        elif dataset_json['name'] == 'MMWHS':
            model = m2ftrans_mmwhs.Model(num_cls = len(dataset_json['labels']))
        else:
            raise NotImplementedError
        return model
    
    def configure_optimizers(self):
        optimizer = AdamW(self.network.parameters(),
                          lr = self.initial_lr,
                          weight_decay = self.weight_decay,
                          amsgrad = True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)

        return optimizer, lr_scheduler
    
    def _build_loss(self):
        class Loss(nn.Module): 
            def __init__(self):
                super().__init__()
            def forward(self, fuse_pred, sep_preds, prm_preds, target, num_cls):
                fuse_cross_loss = criterions.softmax_weighted_loss(fuse_pred, target, num_cls=num_cls)
                fuse_dice_loss = criterions.dice_loss(fuse_pred, target, num_cls=num_cls)
                fuse_loss = fuse_cross_loss + fuse_dice_loss

                sep_cross_loss = torch.zeros(1).cuda().float()
                sep_dice_loss = torch.zeros(1).cuda().float()
                for sep_pred in sep_preds:
                    sep_cross_loss += criterions.softmax_weighted_loss(sep_pred, target, num_cls=num_cls)
                    sep_dice_loss += criterions.dice_loss(sep_pred, target, num_cls=num_cls)
                sep_loss = sep_cross_loss + sep_dice_loss

                prm_cross_loss = torch.zeros(1).cuda().float()
                prm_dice_loss = torch.zeros(1).cuda().float()
                for prm_pred in prm_preds:
                    prm_cross_loss += criterions.softmax_weighted_loss(prm_pred, target, num_cls=num_cls)
                    prm_dice_loss += criterions.dice_loss(prm_pred, target, num_cls=num_cls)
                prm_loss = prm_cross_loss + prm_dice_loss

                loss= fuse_loss + sep_loss + prm_loss

                return loss
        return Loss()
