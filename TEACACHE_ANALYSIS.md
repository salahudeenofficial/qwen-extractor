# TeaCache Implementation for Qwen Image Edit Model

## Overview

TeaCache (Timestep Embedding Aware Cache) is a training-free caching mechanism
that speeds up diffusion model inference by:
1. Comparing timestep embeddings between steps
2. Reusing transformer outputs when embeddings are similar
3. Skipping redundant transformer block computations

## How TeaCache Works (from LightX2V Wan implementation)

### Key Components:

```python
class WanTransformerInferTeaCaching:
    def __init__(self, config):
        self.teacache_thresh = config["teacache_thresh"]  # Similarity threshold
        self.accumulated_rel_l1_distance = 0              # Accumulated difference
        self.previous_e0 = None                           # Previous timestep embedding
        self.previous_residual = None                     # Cached residual (output - input)
        self.coefficients = [...]                         # Rescaling coefficients
        self.ret_steps = 5                                # First N steps always calculate
        self.cutoff_steps = infer_steps - 1               # Last steps always calculate
```

### Algorithm:

1. **Always calculate** for first `ret_steps` and last step
2. **For middle steps:**
   - Calculate L1 distance between current and previous timestep embedding
   - Apply polynomial rescaling: `rescale_func(distance)`
   - Accumulate the rescaled distance
   - If `accumulated_distance < threshold`: **use cache**
   - If `accumulated_distance >= threshold`: **recalculate** and reset accumulator

3. **Cache mechanism:**
   - When calculating: Store `residual = output - input`
   - When using cache: `output = input + cached_residual`

## Implementation Plan for Qwen Image Model

### Step 1: Create TeaCaching Transformer Infer class

File: `lightx2v/models/networks/qwen_image/infer/teacache_transformer_infer.py`

This class extends `QwenImageTransformerInfer` and adds:
- Timestep embedding comparison logic
- Cache storage for residuals
- Decision logic for using cache vs calculating

### Step 2: Modify model.py to use TeaCaching class

When `config["feature_caching"] == "Tea"`:
- Use `QwenImageTeaCachingTransformerInfer` instead of `QwenImageTransformerInfer`

### Step 3: Add coefficients for Qwen Image model

The rescaling coefficients need to be calibrated for the Qwen Image model.
Default Wan coefficients can be used as starting point, then tuned.

## Key Code to Implement

### Main Logic (adapted from Wan):

```python
class QwenImageTeaCachingTransformerInfer(QwenImageTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        self.teacache_thresh = config.get("teacache_thresh", 0.15)
        self.accumulated_rel_l1_distance = 0
        self.previous_embed = None
        self.previous_residual = None
        # For 4-step distillation: always calc first and last step
        self.ret_steps = 1
        self.cutoff_steps = config["infer_steps"] - 1
        # Rescaling coefficients (need calibration)
        self.coefficients = [1.0]  # Linear scaling as default
    
    def calculate_should_calc(self, embed):
        """Decide whether to calculate or use cache."""
        step_index = self.scheduler.step_index
        
        # Always calculate first and last steps
        if step_index < self.ret_steps or step_index >= self.cutoff_steps:
            self.accumulated_rel_l1_distance = 0
            return True
        
        # Calculate L1 distance
        if self.previous_embed is not None:
            rel_l1 = (embed - self.previous_embed).abs().mean() / self.previous_embed.abs().mean()
            rescale_func = np.poly1d(self.coefficients)
            self.accumulated_rel_l1_distance += rescale_func(rel_l1.cpu().item())
        
        # Threshold check
        if self.accumulated_rel_l1_distance < self.teacache_thresh:
            should_calc = False
        else:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        
        self.previous_embed = embed.clone()
        return should_calc
    
    def infer(self, block_weights, pre_infer_out):
        """Override to add caching logic."""
        should_calc = self.calculate_should_calc(pre_infer_out.temb_img_silu)
        
        if should_calc:
            # Full calculation
            ori_hidden = pre_infer_out.hidden_states.clone()
            result = super().infer(block_weights, pre_infer_out)
            self.previous_residual = result - ori_hidden
            return result
        else:
            # Use cache
            return pre_infer_out.hidden_states + self.previous_residual
```

## Implementation Complexity

| Component | Complexity | Notes |
|-----------|------------|-------|
| TeaCache class | Medium | ~200 lines, follows Wan pattern |
| Model.py changes | Low | ~10 lines, class selection |
| Coefficient tuning | Medium | Requires experimentation |
| Testing | Medium | Need to compare quality |

## Expected Speedup

For 4-step distillation:
- Step 0: Always calculate
- Step 1: Maybe skip (depends on threshold)
- Step 2: Maybe skip
- Step 3: Always calculate (last step)

Potential: **Skip 1-2 transformer passes → 25-50% speedup**

## Risks

1. Quality degradation if threshold too high
2. Different coefficients needed for image vs video models
3. CFG (conditional flow guidance) complicates caching if enabled

## Next Steps

1. Create `teacache_transformer_infer.py` for Qwen Image
2. Modify `model.py` to use it when feature_caching="Tea"
3. Test with default threshold (0.15-0.26)
4. Tune coefficients for quality/speed tradeoff
