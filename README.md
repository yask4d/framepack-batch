# FramePack-F1 Video Generation Tool

## Description

FramePack-F1 is an advanced tool for generating videos from static images using artificial intelligence. The system employs diffusion models to create smooth animations based on an input image and a descriptive prompt.

## Key Features

- Video generation from static images
- Support for single and batch processing
- Intuitive graphical interface with Gradio
- Precise control over generation parameters
- Memory optimization for GPUs with different capabilities

## Requirements

- Python 3.8+
- PyTorch with CUDA support
- Dependencies listed in `requirements.txt`
- CUDA-compatible GPU (recommended)

## Installation

1. Clone the repository:
   ```bash
   git clone [repository URL]
   cd [repository name]
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set environment variables (optional):
   ```bash
   export HF_HOME=./hf_download
   ```

## Usage

### Graphical Interface

Run the graphical interface with:
```bash
python batch.py
```

Available options:
- `--share`: Enables remote access
- `--server`: Specifies server IP address
- `--port`: Specifies port number
- `--inbrowser`: Automatically opens in browser

### Batch Mode

For batch processing:
```bash
python batch.py --batch --input_folder [input_folder] --output_folder [output_folder] --duration [duration] --seed [seed] --steps [steps]
```

### Batch Processing Structure

1. Create a folder with input images (.png, .jpg, .jpeg)
2. Add a file named 'prompt' with one prompt per line
3. Optional: Create 'batch_params.txt' to override settings

## Important Parameters

- **Seed**: Random generation seed
- **Total Video Length**: Video duration in seconds
- **Steps**: Number of inference steps (not recommended to change)
- **Distilled CFG Scale**: Controls adherence to prompt
- **GPU Memory Preservation**: Memory to reserve for other processes
- **MP4 Compression**: Output video quality (0=best quality)

## Prompt Examples

- "The girl dances gracefully, with clear movements, full of charm."
- "A character doing some simple body movements."

## Support and Community

Share your results and find ideas on the [FramePack Twitter (X) thread](https://x.com/search?q=framepack&f=live)

## Notes

- For GPUs with less than 60GB VRAM, High-VRAM mode is automatically disabled
- Quality may vary depending on parameters used
- Processing can be resource-intensive for long videos

## License

Apache-2.0 license

# Cite

    @article{zhang2025framepack,
        title={Packing Input Frame Contexts in Next-Frame Prediction Models for Video Generation},
        author={Lvmin Zhang and Maneesh Agrawala},
        journal={Arxiv},
        year={2025}
    }

