"""
TeaCache Transformer Infer for Qwen Image Model

This module implements TeaCache (Timestep Embedding Aware Cache) for the
Qwen Image Edit model in LightX2V. It speeds up inference by caching and
reusing transformer outputs when timestep embeddings are similar.

Usage:
    When config["feature_caching"] == "Tea", this class is used instead of
    the standard QwenImageTransformerInfer.
"""

import gc
import numpy as np
import torch

from lightx2v.models.networks.qwen_image.infer.transformer_infer import QwenImageTransformerInfer


class QwenImageTeaCacheTransformerInfer(QwenImageTransformerInfer):
    """
    TeaCache implementation for Qwen Image model.
    
    Caches transformer outputs and reuses them when timestep embeddings
    are similar enough (below threshold).
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # TeaCache parameters
        self.teacache_thresh = config.get("teacache_thresh", 0.15)
        
        # Accumulated L1 distance
        self.accumulated_rel_l1_distance = 0
        
        # Previous timestep embedding for comparison
        self.previous_embed = None
        
        # Cached residual (output - input)
        self.previous_residual = None
        
        # Rescaling coefficients - calibrated for image generation
        # For 4-step distillation, linear scaling works well
        self.coefficients = config.get("coefficients", [1.0])
        
        # Steps to always calculate (first and last)
        # For 4-step: steps 0,1,2,3 -> always calc 0 and 3
        self.use_ret_steps = config.get("use_ret_steps", False)
        if self.use_ret_steps:
            self.ret_steps = 2  # Calculate first 2 steps
            self.cutoff_steps = config["infer_steps"]  # All last steps
        else:
            self.ret_steps = 1  # Calculate first step only
            self.cutoff_steps = config["infer_steps"] - 1  # Calculate last step
        
        # Statistics for debugging
        self.cache_hits = 0
        self.cache_misses = 0
    
    def calculate_should_calc(self, embed):
        """
        Determine whether to calculate or use cache based on timestep embedding.
        
        Args:
            embed: Current timestep embedding (temb_img_silu)
            
        Returns:
            bool: True if should calculate, False if should use cache
        """
        step_index = self.scheduler.step_index
        
        # Always calculate for first steps
        if step_index < self.ret_steps:
            self.accumulated_rel_l1_distance = 0
            self.cache_misses += 1
            return True
        
        # Always calculate for last step(s)
        if step_index >= self.cutoff_steps:
            self.accumulated_rel_l1_distance = 0
            self.cache_misses += 1
            return True
        
        # Calculate L1 distance if we have a previous embedding
        if self.previous_embed is not None:
            # Relative L1 distance
            with torch.no_grad():
                rel_l1 = (
                    (embed - self.previous_embed).abs().mean() / 
                    (self.previous_embed.abs().mean() + 1e-8)
                )
            
            # Apply polynomial rescaling
            rescale_func = np.poly1d(self.coefficients)
            self.accumulated_rel_l1_distance += rescale_func(rel_l1.cpu().item())
        
        # Threshold check
        if self.accumulated_rel_l1_distance < self.teacache_thresh:
            should_calc = False
            self.cache_hits += 1
        else:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
            self.cache_misses += 1
        
        # Store current embedding for next comparison
        self.previous_embed = embed.detach().clone()
        
        return should_calc
    
    def infer(self, block_weights, pre_infer_out):
        """
        Override of base infer method to add TeaCache logic.
        
        If calculate_should_calc returns True:
          - Run full transformer inference
          - Cache the residual (output - input)
          
        If calculate_should_calc returns False:
          - Reuse cached residual (add to input)
        """
        hidden_states = pre_infer_out.hidden_states
        encoder_hidden_states = pre_infer_out.encoder_hidden_states
        temb_img_silu = pre_infer_out.temb_img_silu
        temb_txt_silu = pre_infer_out.temb_txt_silu
        image_rotary_emb = pre_infer_out.image_rotary_emb
        
        # Decide whether to calculate or use cache
        should_calc = self.calculate_should_calc(temb_img_silu)
        
        if should_calc:
            # Full calculation
            original_hidden = hidden_states.clone()
            
            result = self.infer_func(
                block_weights.blocks,
                hidden_states,
                encoder_hidden_states,
                temb_img_silu,
                temb_txt_silu,
                image_rotary_emb,
                self.scheduler.modulate_index,
            )
            
            # Cache the residual
            self.previous_residual = result - original_hidden
            
            return result
        else:
            # Use cached residual
            if self.previous_residual is not None:
                return hidden_states + self.previous_residual
            else:
                # Fallback to full calculation if no cache
                return self.infer_func(
                    block_weights.blocks,
                    hidden_states,
                    encoder_hidden_states,
                    temb_img_silu,
                    temb_txt_silu,
                    image_rotary_emb,
                    self.scheduler.modulate_index,
                )
    
    def get_cache_stats(self):
        """Return cache hit/miss statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
        }
    
    def clear(self):
        """Clear cached data to free memory."""
        if self.previous_embed is not None:
            del self.previous_embed
            self.previous_embed = None
        if self.previous_residual is not None:
            del self.previous_residual
            self.previous_residual = None
        self.accumulated_rel_l1_distance = 0
        self.cache_hits = 0
        self.cache_misses = 0
        torch.cuda.empty_cache()
        gc.collect()
