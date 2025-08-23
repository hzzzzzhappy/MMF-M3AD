import copy
import random
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from models.initializer import initialize_from_cfg
from torch import Tensor, nn

class MultiScaleFeatureFusion(nn.Module):
   def __init__(self, channels, feature_size):
       super(MultiScaleFeatureFusion, self).__init__()
       self.feature_size = feature_size
       self.downsample_2x = nn.Sequential(nn.Linear(channels, channels), nn.ReLU(inplace=True), nn.Linear(channels, channels))
       self.downsample_4x = nn.Sequential(nn.Linear(channels, channels), nn.ReLU(inplace=True), nn.Linear(channels, channels))
       self.upsample_2x = nn.Sequential(nn.Linear(channels, channels), nn.ReLU(inplace=True), nn.Linear(channels, channels))
       self.upsample_4x = nn.Sequential(nn.Linear(channels, channels), nn.ReLU(inplace=True), nn.Linear(channels, channels))
       self.scale_attention = nn.MultiheadAttention(channels, num_heads=8, dropout=0.1)
       self.fusion_conv = nn.Sequential(nn.Linear(channels * 3, channels), nn.ReLU(inplace=True), nn.Linear(channels, channels))
       self.scale_weights = nn.Parameter(torch.ones(3) / 3)

   def downsample_features(self, x, scale):
       G, B, C = x.shape
       if scale == 2:
           downsampled = x.view(G//2, 2, B, C).mean(dim=1)
           downsampled = self.downsample_2x(downsampled)
       elif scale == 4:
           downsampled = x.view(G//4, 4, B, C).mean(dim=1)
           downsampled = self.downsample_4x(downsampled)
       return downsampled

   def upsample_features(self, x, target_size, scale):
       G, B, C = x.shape
       if scale == 2:
           upsampled = x.repeat_interleave(2, dim=0)[:target_size]
           upsampled = self.upsample_2x(upsampled)
       elif scale == 4:
           upsampled = x.repeat_interleave(4, dim=0)[:target_size]
           upsampled = self.upsample_4x(upsampled)
       return upsampled

   def cross_scale_attention(self, features_list):
       all_features = torch.cat(features_list, dim=0)
       attended_features, _ = self.scale_attention(all_features, all_features, all_features)
       G = features_list[0].shape[0]
       scale1_out = attended_features[:G]
       scale2_out = attended_features[G:G+G//2]
       scale3_out = attended_features[G+G//2:]
       return [scale1_out, scale2_out, scale3_out]

   def forward(self, x):
       G, B, C = x.shape
       scale1_features = x
       scale2_features = self.downsample_features(x, scale=2)
       scale3_features = self.downsample_features(x, scale=4)
       attended_scales = self.cross_scale_attention([scale1_features, scale2_features, scale3_features])
       scale1_final = attended_scales[0]
       scale2_final = self.upsample_features(attended_scales[1], G, scale=2)
       scale3_final = self.upsample_features(attended_scales[2], G, scale=4)
       concatenated = torch.cat([scale1_final, scale2_final, scale3_final], dim=-1)
       fused_features = self.fusion_conv(concatenated)
       weights = F.softmax(self.scale_weights, dim=0)
       weighted_fusion = (weights[0] * scale1_final + weights[1] * scale2_final + weights[2] * scale3_final)
       final_output = fused_features + weighted_fusion
       return final_output

class SCFA(nn.Module):
   def __init__(self, channels, h_kernel_size=11, v_kernel_size=11, reduction=8):
       super(SCFA, self).__init__()
       self.avg_pool = nn.AdaptiveAvgPool1d(1)
       self.caa_conv1 = nn.Sequential(nn.Conv1d(channels, channels//reduction, 1, bias=True), nn.ReLU(inplace=True))
       self.h_conv = nn.Conv1d(channels//reduction, channels//reduction, h_kernel_size, 1, h_kernel_size//2, groups=channels//reduction)
       self.v_conv = nn.Conv1d(channels//reduction, channels//reduction, v_kernel_size, 1, v_kernel_size//2, groups=channels//reduction)
       self.caa_conv2 = nn.Sequential(nn.Conv1d(channels//reduction, channels, 1, bias=True), nn.Sigmoid())
       self.fa_ca = nn.Sequential(nn.Linear(channels, channels // reduction), nn.ReLU(inplace=True), nn.Linear(channels // reduction, channels), nn.Sigmoid())
       self.fa_ga = nn.Sequential(nn.Linear(channels, channels // reduction), nn.ReLU(inplace=True), nn.Linear(channels // reduction, 1), nn.Sigmoid())
       self.fusion_weight = nn.Parameter(torch.tensor(0.5))
       self.multi_scale_fusion = MultiScaleFeatureFusion(channels, feature_size=64)

   def forward(self, x):
       x = self.multi_scale_fusion(x)
       G, B, C = x.shape
       x_reshaped = x.view(G*B, C, 1)
       global_info = self.avg_pool(x_reshaped)
       conv1_out = self.caa_conv1(global_info)
       h_context = self.h_conv(conv1_out)
       v_context = self.v_conv(h_context)
       caa_attn_factor = self.caa_conv2(v_context)
       caa_out = (x_reshaped * caa_attn_factor).view(G, B, C)
       ca_weight = torch.mean(x, dim=0, keepdim=True)
       ca_weight = self.fa_ca(ca_weight)
       x_ca = x * ca_weight
       ga_weight = self.fa_ga(x_ca)
       fa_out = x_ca * ga_weight
       fused = self.fusion_weight * caa_out + (1 - self.fusion_weight) * fa_out
       return fused

class LPRM(nn.Module):
   def __init__(self, hidden_dim, num_levels=3, expansion_factor=2):
       super(LPRM, self).__init__()
       self.coarse_reconstructor = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * expansion_factor), nn.ReLU(inplace=True), nn.Linear(hidden_dim * expansion_factor, hidden_dim), nn.Dropout(0.1))
       self.medium_reconstructor = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * expansion_factor), nn.GELU(), nn.Linear(hidden_dim * expansion_factor, hidden_dim), nn.Dropout(0.1))
       self.fine_reconstructor = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * expansion_factor), nn.SiLU(), nn.Linear(hidden_dim * expansion_factor, hidden_dim), nn.Dropout(0.1))
       self.level_weights = nn.Parameter(torch.ones(num_levels) / num_levels)
       self.quality_gate = nn.Sequential(nn.Linear(hidden_dim * num_levels, hidden_dim), nn.Sigmoid())

   def forward(self, x):
       coarse_out = self.coarse_reconstructor(x)
       medium_out = self.medium_reconstructor(coarse_out + x)
       fine_out = self.fine_reconstructor(medium_out + x)
       level_outputs = torch.stack([coarse_out, medium_out, fine_out], dim=-1)
       weights = F.softmax(self.level_weights, dim=0)
       weighted_output = torch.sum(level_outputs * weights, dim=-1)
       concatenated = torch.cat([coarse_out, medium_out, fine_out], dim=-1)
       quality_weights = self.quality_gate(concatenated)
       final_output = weighted_output * quality_weights + x * (1 - quality_weights)
       return final_output

class UniAD(nn.Module):
   def __init__(self, feature_size, feature_jitter, neighbor_mask, hidden_dim, initializer, cls_num, inplanes=1152, k=5, mask_ratio=0.4, **kwargs):
       super().__init__()
       self.feature_jitter = feature_jitter
       self.cls_num = cls_num
       self.pos_embed = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, hidden_dim))
       self.distence = torch.nn.MSELoss()
       self.transformer = Transformer(hidden_dim, feature_size, neighbor_mask, **kwargs)
       self.input_proj = nn.Linear(inplanes, hidden_dim)
       self.output_proj = nn.Linear(hidden_dim, inplanes)
       self.cls_head_finetune = nn.Sequential(nn.Linear(inplanes*2, 256), nn.LayerNorm(256), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(256, self.cls_num))
       self.mask_ratio = mask_ratio
       initialize_from_cfg(self, initializer)

   def add_jitter(self, feature_tokens, scale, prob):
       if random.uniform(0, 1) <= prob:
           num_tokens, batch_size, dim_channel = feature_tokens.shape
           feature_norms = (feature_tokens.norm(dim=2).unsqueeze(2) / dim_channel)
           jitter = torch.randn((num_tokens, batch_size, dim_channel)).cuda()
           jitter = jitter * feature_norms * scale
           feature_tokens = feature_tokens + jitter
       return feature_tokens

   def forward(self, input):
       feature_align = input["xyz_features"]
       center = input["center"]
       geome_vars = torch.zeros(center.shape[0], center.shape[1]).cuda()
       feature_tokens = rearrange(feature_align, "b n g -> g b n")
       if self.training and self.feature_jitter:
           feature_tokens = self.add_jitter(feature_tokens, self.feature_jitter.scale, self.feature_jitter.prob)
       feature_tokens = self.input_proj(feature_tokens)
       pos_embed = self.pos_embed(center).permute(1,0,2)
       output_decoder, _ = self.transformer(feature_tokens, pos_embed ,geome_vars,self.mask_ratio)
       feature_rec_tokens = self.output_proj(output_decoder)
       feature_rec = rearrange(feature_rec_tokens, "g b n -> b n g")
       feature_cls = feature_rec.detach().clone()
       feature_cls.requires_grad = True
       feature_cls = rearrange(feature_cls,"b n g -> b g n")
       concat_f = torch.cat([feature_cls[:, 0], feature_cls[:, 1:].max(1)[0]], dim=-1)
       cls_pred = self.cls_head_finetune(concat_f)
       pred = torch.sqrt(torch.sum((feature_rec - feature_align) ** 2, dim=1, keepdim=True))
       return {"feature_rec": feature_rec, "feature_align": feature_align, "pred": pred, "cls_pred": cls_pred}

class Transformer(nn.Module):
   def __init__(self, hidden_dim, feature_size, neighbor_mask, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1, activation="relu", normalize_before=False, return_intermediate_dec=False):
       super().__init__()
       self.feature_size = feature_size
       self.neighbor_mask = neighbor_mask
       encoder_layer = TransformerEncoderLayerWithSCFA(hidden_dim, nhead, dim_feedforward, dropout, activation, normalize_before)
       encoder_norm = nn.LayerNorm(hidden_dim) if normalize_before else None
       self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
       decoder_layer = TransformerDecoderLayerWithLPRM(hidden_dim, feature_size, nhead, dim_feedforward, dropout, activation, normalize_before)
       decoder_norm = nn.LayerNorm(hidden_dim)
       self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec)
       self.hidden_dim = hidden_dim
       self.nhead = nhead

   def generate_mask(self, feature_size, geome_vars,mask_ratio=0.4):
       B = geome_vars.shape[0]
       mask = torch.zeros((B, feature_size), dtype=torch.bool)
       for idx in range(B):
           indices = torch.randperm(feature_size)
           top_k = int(feature_size * mask_ratio)
           mask_indices = indices[:top_k]
           mask[idx, mask_indices] = True
       mask = mask.cuda()
       return mask

   def forward(self, src, pos_embed, geome_vars,mask_ratio):
       if self.neighbor_mask:
           mask = self.generate_mask(self.feature_size, geome_vars,mask_ratio)
           mask_enc = mask if self.neighbor_mask.mask[0] else None
           mask_dec1 = mask if self.neighbor_mask.mask[1] else None
           mask_dec2 = mask if self.neighbor_mask.mask[2] else None
       else:
           mask_enc = mask_dec1 = mask_dec2 = None
       output_encoder = self.encoder(src, src_key_padding_mask=mask_enc, pos=pos_embed)
       output_decoder = self.decoder(output_encoder, tgt_key_padding_mask=mask_dec1, pos=pos_embed)
       return output_decoder, output_encoder

class TransformerEncoder(nn.Module):
   def __init__(self, encoder_layer, num_layers, norm=None):
       super().__init__()
       self.layers = _get_clones(encoder_layer, num_layers)
       self.num_layers = num_layers
       self.norm = norm

   def forward(self, src, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None):
       output = src
       for layer in self.layers:
           output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
       if self.norm is not None:
           output = self.norm(output)
       return output

class TransformerDecoder(nn.Module):
   def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
       super().__init__()
       self.layers = _get_clones(decoder_layer, num_layers)
       self.num_layers = num_layers
       self.norm = norm
       self.return_intermediate = return_intermediate

   def forward(self, memory, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None):
       output = memory
       intermediate = []
       for layer in self.layers:
           output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask, pos=pos)
           if self.return_intermediate:
               intermediate.append(self.norm(output))
       if self.norm is not None:
           output = self.norm(output)
           if self.return_intermediate:
               intermediate.pop()
               intermediate.append(output)
       if self.return_intermediate:
           return torch.stack(intermediate)
       return output

class TransformerEncoderLayerWithSCFA(nn.Module):
   def __init__(self, hidden_dim, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
       super().__init__()
       self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
       self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
       self.dropout = nn.Dropout(dropout)
       self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
       self.norm1 = nn.LayerNorm(hidden_dim)
       self.norm2 = nn.LayerNorm(hidden_dim)
       self.norm3 = nn.LayerNorm(hidden_dim)
       self.dropout1 = nn.Dropout(dropout)
       self.dropout2 = nn.Dropout(dropout)
       self.dropout3 = nn.Dropout(dropout)
       self.scfa = SCFA(hidden_dim)
       self.activation = _get_activation_fn(activation)
       self.normalize_before = normalize_before

   def with_pos_embed(self, tensor, pos):
       return tensor if pos is None else tensor + pos

   def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
       q = k = self.with_pos_embed(src, pos)
       src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
       src = src + self.dropout1(src2)
       src = self.norm1(src)
       src2 = self.scfa(src)
       src = src + self.dropout3(src2)
       src = self.norm3(src)
       src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
       src = src + self.dropout2(src2)
       src = self.norm2(src)
       return src

   def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
       src2 = self.norm1(src)
       q = k = self.with_pos_embed(src2, pos)
       src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
       src = src + self.dropout1(src2)
       src2 = self.norm3(src)
       src2 = self.scfa(src2)
       src = src + self.dropout3(src2)
       src2 = self.norm2(src)
       src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
       src = src + self.dropout2(src2)
       return src

   def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
       if self.normalize_before:
           return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
       return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class TransformerDecoderLayerWithLPRM(nn.Module):
   def __init__(self, hidden_dim, feature_size, nhead, dim_feedforward, dropout=0.1, activation="relu", normalize_before=False):
       super().__init__()
       num_queries = feature_size
       self.learned_embed = nn.Embedding(num_queries, hidden_dim)
       self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
       self.multihead_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
       self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
       self.dropout = nn.Dropout(dropout)
       self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
       self.norm1 = nn.LayerNorm(hidden_dim)
       self.norm2 = nn.LayerNorm(hidden_dim)
       self.norm3 = nn.LayerNorm(hidden_dim)
       self.norm4 = nn.LayerNorm(hidden_dim)
       self.dropout1 = nn.Dropout(dropout)
       self.dropout2 = nn.Dropout(dropout)
       self.dropout3 = nn.Dropout(dropout)
       self.dropout4 = nn.Dropout(dropout)
       self.lprm = LPRM(hidden_dim)
       self.activation = _get_activation_fn(activation)
       self.normalize_before = normalize_before

   def with_pos_embed(self, tensor, pos: Optional[Tensor]):
       return tensor if pos is None else tensor + pos

   def forward_post(self, out, memory, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None):
       tgt = pos
       tgt2 = self.self_attn(query=tgt, key=self.with_pos_embed(memory, pos), value=memory, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
       tgt = tgt + self.dropout1(tgt2)
       tgt = self.norm1(tgt)
       tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, pos), key=self.with_pos_embed(out, pos), value=out, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
       tgt = tgt + self.dropout2(tgt2)
       tgt = self.norm2(tgt)
       tgt2 = self.lprm(tgt)
       tgt = tgt + self.dropout4(tgt2)
       tgt = self.norm4(tgt)
       tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
       tgt = tgt + self.dropout3(tgt2)
       tgt = self.norm3(tgt)
       return tgt

   def forward_pre(self, out, memory, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None):
       tgt = pos
       tgt2 = self.norm1(tgt)
       tgt2 = self.self_attn(query=self.with_pos_embed(tgt2, pos), key=self.with_pos_embed(memory, pos), value=memory, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
       tgt = tgt + self.dropout1(tgt2)
       tgt2 = self.norm2(tgt)
       tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, pos), key=self.with_pos_embed(out, pos), value=out, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
       tgt = tgt + self.dropout2(tgt2)
       tgt2 = self.norm4(tgt)
       tgt2 = self.lprm(tgt2)
       tgt = tgt + self.dropout4(tgt2)
       tgt2 = self.norm3(tgt)
       tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
       tgt = tgt + self.dropout3(tgt2)
       return tgt

   def forward(self, out, memory, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None):
       if self.normalize_before:
           return self.forward_pre(out, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos)
       return self.forward_post(out, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos)

def _get_clones(module, N):
   return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
   if activation == "relu":
       return F.relu
   if activation == "gelu":
       return F.gelu
   if activation == "glu":
       return F.glu
   raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
