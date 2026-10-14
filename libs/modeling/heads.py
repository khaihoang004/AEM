import math
import torch
from torch import nn
from torch.nn import functional as F

from .blocks import MaskedConv1D, Scale, LayerNorm, DynamicFeatureAttentionLayer


    
class CrossScaleLayer(nn.Module):
    """Simplified cross-scale information exchange layer with weighted fusion"""
    
    def __init__(self, in_dim, out_dim, kernel_size=3, with_ln=False, num_adjacent_scales=2, depth_module=None):
        super().__init__()
        self.num_adjacent_scales = num_adjacent_scales
        
        # Individual processing conv
        self.individual_conv = MaskedConv1D(
            in_dim, out_dim, kernel_size,
            stride=1, padding=kernel_size//2, bias=(not with_ln)
        )
        self.norm = LayerNorm(out_dim) if with_ln else nn.Identity()
        
        # Weighted fusion parameters
        self.scale_weight = nn.Parameter(torch.zeros(1))
        self.output_weight = nn.Parameter(torch.ones(1))
        
        # Resize function using interpolation
        self.resize = lambda x, s: F.interpolate(x, size=s, mode="linear", align_corners=False)
        self.depth_module = depth_module
    
    def forward(self, fpn_feats, fpn_masks):
        # Step 1: Process each level individually
        processed_feats = []
        for feat, mask in zip(fpn_feats, fpn_masks):
            processed_feat, _ = self.individual_conv(feat, mask)
            processed_feat = self.norm(processed_feat)
            processed_feats.append(processed_feat)
        
        # Step 2: Weighted fusion across scales
        enhanced_feats = []
        
        for l, current_feat in enumerate(processed_feats):
            scale_features = []
            for s in range(self.num_adjacent_scales):
                l_source = l + s - self.num_adjacent_scales // 2
                l_source = l_source if l_source < l else l_source + 1
                
                if 0 <= l_source < len(processed_feats):
                    feature = self.resize(processed_feats[l_source], current_feat.shape[-1:])
                    scale_features.append(feature)
            
            if scale_features:
                fused_feature = sum(scale_features)
                enhanced_feat = fused_feature * self.scale_weight + current_feat * self.output_weight
            else:
                enhanced_feat = current_feat
            
            if self.depth_module is not None:
                enhanced_feat, masks = self.depth_module(enhanced_feat, fpn_masks[l])
            
            enhanced_feats.append(enhanced_feat)
        
        return enhanced_feats, fpn_masks


class DynClsHead(nn.Module):
    """Simplified dynamic multi-scale classification head with weighted fusion"""
    
    def __init__(self, input_dim, feat_dim, num_classes, prior_prob=0.01,
                 num_layers=3, kernel_size=3, act_layer=nn.ReLU, 
                 with_ln=False, empty_cls=[], num_adjacent_scales=2):
        super().__init__()
        self.act = act_layer()
        self.head = nn.ModuleList()
        
        for idx in range(num_layers - 1):
            in_dim = input_dim if idx == 0 else feat_dim
            depth_module = DynamicFeatureAttentionLayer(dim=feat_dim)
            layer = CrossScaleLayer(
                in_dim, feat_dim, kernel_size, 
                with_ln=with_ln, num_adjacent_scales=num_adjacent_scales,
                depth_module=depth_module
            )
            self.head.append(layer)
        
        # Classification head
        self.cls_head = MaskedConv1D(
            feat_dim, num_classes, kernel_size,
            stride=1, padding=kernel_size//2
        )
        
        # Initialize bias
        if prior_prob > 0:
            bias_value = -(math.log((1 - prior_prob) / prior_prob))
            torch.nn.init.constant_(self.cls_head.conv.bias, bias_value)
        
    def forward(self, fpn_feats, fpn_masks):
        """Forward pass with weighted multi-scale fusion"""
        feats = fpn_feats
        masks = fpn_masks
        
        # Apply cross-scale layers
        for layer in self.head:
            feats, masks = layer(feats, masks)
            feats = [self.act(feat) for feat in feats]
        
        # Classification output
        out_logits = tuple()
        for cur_feat, cur_mask in zip(feats, masks):
            cur_logits, _ = self.cls_head(cur_feat, cur_mask)
            out_logits += (cur_logits,)
        
        return out_logits


class DynRegHead(nn.Module):
    """Simplified dynamic multi-scale regression head with weighted fusion"""
    
    def __init__(self, input_dim, feat_dim, fpn_levels, num_layers=3,
                 kernel_size=3, act_layer=nn.ReLU, with_ln=False, num_adjacent_scales=2):
        super().__init__()
        self.fpn_levels = fpn_levels
        self.act = act_layer()
        self.head = nn.ModuleList()
        
        for idx in range(num_layers - 1):
            in_dim = input_dim if idx == 0 else feat_dim
            depth_module = DynamicFeatureAttentionLayer(dim=feat_dim)
            layer = CrossScaleLayer(
                in_dim, feat_dim, kernel_size,
                with_ln=with_ln, num_adjacent_scales=num_adjacent_scales,
                depth_module=depth_module
            )
            self.head.append(layer)
        
        # Regression components
        self.scale = nn.ModuleList([Scale() for _ in range(fpn_levels)])
        self.offset_head = MaskedConv1D(
            feat_dim, 2, kernel_size, stride=1, padding=kernel_size//2
        )
    
    def forward(self, fpn_feats, fpn_masks):
        """Forward pass with weighted multi-scale fusion"""
        assert len(fpn_feats) == len(fpn_masks) == self.fpn_levels
        
        feats = fpn_feats
        masks = fpn_masks
        
        # Apply cross-scale layers
        for layer in self.head:
            feats, masks = layer(feats, masks)
            feats = [self.act(feat) for feat in feats]
        
        # Regression output
        out_offsets = tuple()
        for l, (cur_feat, cur_mask) in enumerate(zip(feats, masks)):
            cur_offsets, _ = self.offset_head(cur_feat, cur_mask)
            out_offsets += (F.relu(self.scale[l](cur_offsets)),)
        
        return out_offsets