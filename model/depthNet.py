import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.midas.midas_net_custom import MidasNet_small_videpth
from .modules.DFV import DFFNet
from .modules.GlobalScaleEstimator import LeastSquaresEstimator
from .modules.depth_anything.dpt import DepthAnything
from .modules.Attention import TransformerFusion
from .modules.CrossAttention import DepthFusionCrossViT


class DepthNet(nn.Module):
    """A baseline model for depth from focus (DFF)."""

    def __init__(self, invert_depth: bool = True):
        super(DepthNet, self).__init__()
        self.invert_depth = invert_depth
        self.depth_anything = DepthAnything.from_pretrained(
            'LiheYoung/depth_anything_{:}14'.format('vits')).eval()
        self.depth_anything.requires_grad_(False)
        self.depth_anything.cuda()
        self.depth_anything.eval()

        self.DFF_model = DFFNet(clean=False, level=4, use_diff=1)
        self.DFF_model.cuda()
        # DFV_weights = None  # Don't need this for evaluation, only add for training
        DFV_weights = "/mnt/sda1/zg_workspace/DFF-DFV.tar"

        if DFV_weights is not None:
            pretrained_dict = torch.load(DFV_weights)
            pretrained_dict['state_dict'] = {
                k.replace('module.', ''): v
                for k, v in pretrained_dict['state_dict'].items()
                if 'disp' not in k
            }
            self.DFF_model.load_state_dict(
                pretrained_dict['state_dict'], strict=False)
            print('Loaded DFV model')
        else:
            print('Run with random init')

        self.DFF_model.eval()
        self.DFF_model.requires_grad_(False)

        # self.transformer_fusion = TransformerFusion()
        self.depth_fusion = DepthFusionCrossViT()

    def predic_depth_rel(self, batch):
        org_size = batch.shape
        batch = F.interpolate(batch, size=(518, 518), mode='bicubic', align_corners=True)
        with torch.no_grad():
            depth = self.depth_anything(batch).unsqueeze(1)
        depth = F.interpolate(depth, size=(org_size[2], org_size[3]), mode='bicubic', align_corners=True)
        return depth

    def forward(self, rgb_aif, focal_stack, disp_dist):
        org_size = rgb_aif.shape
        self.DFF_model.eval()

        rel_depth = self.predic_depth_rel(rgb_aif)
        pred_dff, std, _ = self.DFF_model(focal_stack, disp_dist)
        if len(pred_dff) == 4:
            pred_dff = pred_dff[3]

        valid_mask = torch.ones((org_size[0], 1, org_size[2], org_size[3]),
                                dtype=torch.bool, device=rgb_aif.device, requires_grad=False)
        if self.invert_depth:
            rel_depth = 1.0 / rel_depth
            rel_depth[rel_depth == float("inf")] = 0

        # metric_depth, _ = self.transformer_fusion(rel_depth, pred_dff, std)
        metric_depth = self.depth_fusion(pred_dff, rel_depth, std)

        return metric_depth, rel_depth, metric_depth, std, pred_dff