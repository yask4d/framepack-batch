# FramePack-F1 (Batch Processing Modification)  
# Video Generation Tool  

## Description  

This version is a modification of the original **[FramePack by lllyasviel](https://github.com/lllyasviel/FramePack)** that adds batch processing functionality and other improvements.  

FramePack-F1 is an advanced tool for generating videos from static images using artificial intelligence. The system uses diffusion models to create smooth animations based on an input image and a descriptive prompt.  

**Key improvements over the original:**  
- Full batch processing support  
- Enhanced Gradio graphical interface  
- Memory optimization for different GPU configurations  
- Simplified setup for users  

## Key Features  

- Video generation from static images  
- **New:** Automatic batch processing  
- Support for individual and batch processing  
- Intuitive Gradio interface  
- Precise control over generation parameters  
- Memory optimization for GPUs with varying capabilities  

## Installation  

1. First install the original FramePack following the instructions at:  
   https://github.com/lllyasviel/FramePack  

2. Then download these two files from our modification:  
   - [run-batch.bat](file_link) - Place it in the FramePack root directory  
   - [batch.py](file_link) - Place it in the FramePack `webui` folder  

3. Execute `run-batch.bat` to launch the enhanced interface  

## Usage  

### Enhanced Graphical Interface  

Run:  
```bash  
run-batch.bat  
```  

### Terminal Mode (new functionality)  

For batch processing:  
```bash  
python batch.py --batch --input_folder [input_folder] --output_folder [output_folder] --duration [duration] --seed [seed] --steps [steps]  
```  

### Batch Processing Structure  

1. Create a folder with input images (.png, .jpg, .jpeg)  
2. Add a file named 'prompt.txt' with one prompt per line  
3. Optional: Create 'batch_params.txt' to override configurations  

## Important Parameters  

- **Seed**: Random generation seed  
- **Total Video Length**: Video duration in seconds  
- **Steps**: Number of inference steps  
- **Distilled CFG Scale**: Controls adherence to prompt  
- **GPU Memory Preservation**: Memory reserved for other processes  
- **MP4 Compression**: Output video quality (0=best quality)  

## Prompt Examples  

- "The girl dances gracefully, with clear movements, full of charm."  
- "A character doing some simple body movements."  

## Support and Community  

Share your results and find ideas in the [FramePack Twitter (X) thread](https://x.com/search?q=framepack&f=live)  

## Notes  

- For GPUs with less than 60GB VRAM, High-VRAM mode is automatically disabled  
- Quality may vary depending on parameters used  
- Batch processing can be resource-intensive for many images  

## Acknowledgments  

This project is based on the original work by lllyasviel:  
https://github.com/lllyasviel/FramePack  

## License  

Apache-2.0 license  

© Kadyr Valdes (YASKAD)  

https://www.youtube.com/c/YasKad  

# Cite  

    @article{zhang2025framepack,  
        title={Packing Input Frame Contexts in Next-Frame Prediction Models for Video Generation},  
        author={Lvmin Zhang and Maneesh Agrawala},  
        journal={Arxiv},  
        year={2025}  
    }