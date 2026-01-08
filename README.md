# LightX2V FP8 Baseline - Virtual Try-On

Fast VTON inference using LightX2V with Qwen-Image-Edit-2511.

## Performance
| Mode | Steps | Time (L40) | VRAM | Output |
|------|-------|------------|------|--------|
| FP8 | 4 | ~12-13s | ~35GB | 768x1024 |
| FP8+TeaCache | 4 | ~10s | ~35GB | 768x1024 |

## Quick Start
See `QUICKSTART.txt` for full instructions.

```bash
# Docker: lightx2v/lightx2v:25101501-cu124

git clone https://github.com/salahudeenofficial/try_og_pipeline.git
cd try_og_pipeline && git checkout fp8-baseline
chmod +x setup_lightx2v.sh && ./setup_lightx2v.sh
python test_lightx2v_vton.py --mode fp8 --person person.jpg --cloth cloth.png
```

## Files
- `test_lightx2v_vton.py` - Main inference script
- `setup_lightx2v.sh` - Setup script
- `person.jpg`, `cloth.png` - Test images
- `QUICKSTART.txt` - Quick start guide
- `PROBLEMS_FACED.txt` - Known issues & fixes
