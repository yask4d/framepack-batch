# Batchfrom diffusers_helper.hf_login import login
import os
os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

import gradio as gr
import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import argparse
import math
import glob
import json
from datetime import datetime
from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket

parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
parser.add_argument("--batch", action='store_true', help="Run in batch mode with input folder")
parser.add_argument("--input_folder", type=str, default='input', help="Folder containing input images and prompt file")
parser.add_argument("--output_folder", type=str, default='outputs', help="Folder to save generated videos")
parser.add_argument("--duration", type=float, default=5.0, help="Total video length in seconds")
parser.add_argument("--seed", type=int, default=31337, help="Seed for video generation")
parser.add_argument("--steps", type=int, default=25, help="Number of sampling steps")
args = parser.parse_args()

# Configuración inicial
print(args)
free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60
print(f'Free VRAM {free_mem_gb} GB')
print(f'High-VRAM Mode: {high_vram}')

# Carga de modelos
text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()

feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePack_F1_I2V_HY_20250503', torch_dtype=torch.bfloat16).cpu()

# Configuración de modelos
vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()
transformer.eval()

if not high_vram:
    vae.enable_slicing()
    vae.enable_tiling()

transformer.high_quality_fp32_output_for_inference = True
print('transformer.high_quality_fp32_output_for_inference = True')

transformer.to(dtype=torch.bfloat16)
vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)
transformer.requires_grad_(False)

if not high_vram:
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    text_encoder.to(gpu)
    text_encoder_2.to(gpu)
    image_encoder.to(gpu)
    vae.to(gpu)
    transformer.to(gpu)

# Stream global para compartir entre funciones
global_stream = AsyncStream()
outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)

# Función para generar seed aleatorio
def generate_random_seed():
    return torch.randint(0, 2**32 - 1, (1,)).item()
    
def clean_previous_outputs(output_folder, job_id=None):
    """Elimina archivos temporales de generaciones anteriores"""
    if job_id:
        # Eliminar archivos específicos de un job_id
        for ext in ['mp4', 'png']:
            pattern = os.path.join(output_folder, f'{job_id}_*.{ext}')
            for temp_file in glob.glob(pattern):
                try:
                    os.remove(temp_file)
                except:
                    pass
    else:
        # Eliminar todos los archivos temporales (_final.mp4 e _input.png se conservan)
        for ext in ['mp4', 'png']:
            pattern = os.path.join(output_folder, f'*_*.{ext}')
            for temp_file in glob.glob(pattern):
                if not ('_final.' in temp_file or '_input.' in temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass

# Función para guardar settings
def save_job_settings(output_folder, job_id, settings):
    """
    Guarda los settings de un job en un archivo JSON
    """
    settings_file = os.path.join(output_folder, f'{job_id}_settings.json')
    settings['timestamp'] = datetime.now().isoformat()
    
    with open(settings_file, 'w') as f:
        json.dump(settings, f, indent=2)

# Función worker
@torch.no_grad()
def worker(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, batch_id=None, output_folder='outputs'):
    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    job_id = generate_timestamp() if batch_id is None else f"batch_{batch_id}"
    
    # Create initial job settings without resolution info
    job_settings = {
        'job_id': job_id,
        'prompt': prompt,
        'negative_prompt': n_prompt,
        'seed': seed,
        'duration_seconds': total_second_length,
        'latent_window_size': latent_window_size,
        'steps': steps,
        'cfg_scale': cfg,
        'distilled_guidance_scale': gs,
        'guidance_rescale': rs,
        'gpu_memory_preservation_gb': gpu_memory_preservation,
        'use_teacache': use_teacache,
        'mp4_crf': mp4_crf,
        'input_image_size': input_image.shape if input_image is not None else None,
        # We'll add output_resolution later after calculating height and width
    }
    
    save_job_settings(output_folder, job_id, job_settings)
    if batch_id is not None:
        global_stream.output_queue.push(('batch_progress', f"Processing batch item {batch_id}..."))
        global_stream.output_queue.push(('progress', (None, f"Starting batch item {batch_id}", make_progress_bar_html(0, 'Starting batch item...'))))
    else:
        global_stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

    # Limpiar archivos previos de la misma generación
    clean_previous_outputs(output_folder, job_id)

    # Variable para el archivo final
    final_output_filename = os.path.join(output_folder, f'{job_id}_final.mp4')
    
    try:
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

        global_stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))

        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)
            load_model_as_complete(text_encoder_2, target_device=gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        global_stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image processing ...'))))

        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        job_settings['output_resolution'] = f"{height}x{width}"
        save_job_settings(output_folder, job_id, job_settings)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

        # Guardar solo la imagen de entrada (sobrescribir si existe)
        Image.fromarray(input_image_np).save(os.path.join(output_folder, f'{job_id}_input.png'))

        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        global_stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))

        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)

        start_latent = vae_encode(input_image_pt, vae)

        global_stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))

        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        global_stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))

        rnd = torch.Generator("cpu").manual_seed(seed)

        history_latents = torch.zeros(size=(1, 16, 16 + 2 + 1, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None

        history_latents = torch.cat([history_latents, start_latent.to(history_latents)], dim=2)
        total_generated_latent_frames = 1

        for section_index in range(total_latent_sections):
            if global_stream.input_queue.top() == 'end':
                global_stream.output_queue.push(('end', None))
                return

            print(f'section_index = {section_index}, total_latent_sections = {total_latent_sections}')

            if not high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            def callback(d):
                preview = d['denoised']
                preview = vae_decode_fake(preview)

                preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                if global_stream.input_queue.top() == 'end':
                    global_stream.output_queue.push(('end', None))
                    raise KeyboardInterrupt('User ends the task.')

                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                hint = f'Sampling {current_step}/{steps}'
                desc = f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30) :.2f} seconds (FPS-30). The video is being extended now ...'
                global_stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
                return

            indices = torch.arange(0, sum([1, 16, 2, 1, latent_window_size])).unsqueeze(0)
            clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = indices.split([1, 16, 2, 1, latent_window_size], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)

            clean_latents_4x, clean_latents_2x, clean_latents_1x = history_latents[:, :, -sum([16, 2, 1]):, :, :].split([16, 2, 1], dim=2)
            clean_latents = torch.cat([start_latent.to(history_latents), clean_latents_1x], dim=2)

            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=latent_window_size * 4 - 3,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=gpu,
                dtype=torch.bfloat16,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([history_latents, generated_latents.to(history_latents)], dim=2)

            if not high_vram:
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)

            real_history_latents = history_latents[:, :, -total_generated_latent_frames:, :, :]

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                section_latent_frames = latent_window_size * 2
                overlapped_frames = latent_window_size * 4 - 3

                current_pixels = vae_decode(real_history_latents[:, :, -section_latent_frames:], vae).cpu()
                history_pixels = soft_append_bcthw(history_pixels, current_pixels, overlapped_frames)

            if not high_vram:
                unload_complete_models()

            # Guardar siempre en el mismo archivo final (sobrescribir)
            save_bcthw_as_mp4(history_pixels, final_output_filename, fps=30, crf=mp4_crf)

            print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')

            global_stream.output_queue.push(('file', final_output_filename))
    except:
        # Eliminar archivos generados si hay error
        if os.path.exists(final_output_filename):
            os.remove(final_output_filename)
        traceback.print_exc()

        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

    global_stream.output_queue.push(('end', None))
    return

def process(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf):
    global global_stream
    assert input_image is not None, 'No input image!'

    yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True)

    # Reiniciar el stream para cada proceso
    global_stream = AsyncStream()

    async_run(worker, input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf)

    output_filename = None

    while True:
        flag, data = global_stream.output_queue.next()

        if flag == 'file':
            output_filename = data
            yield output_filename, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True)

        if flag == 'progress':
            preview, desc, html = data
            yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)
        
        if flag == 'batch_progress':
            message = data
            yield gr.update(), gr.update(visible=False), message, '', gr.update(interactive=False), gr.update(interactive=True)

        if flag == 'end':
            yield output_filename, gr.update(visible=False), gr.update(), '', gr.update(interactive=True), gr.update(interactive=False)
            break

def end_process():
    global global_stream
    
    # Enviar señal de interrupción
    global_stream.input_queue.push('end')
    
    # Descargar todos los modelos de la GPU
    if not high_vram:
        unload_complete_models(
            text_encoder, text_encoder_2, image_encoder, vae, transformer
        )
    
    # Resetear el stream para futuras generaciones
    global_stream = AsyncStream()
    
    # Liberar memoria de CUDA (opcional pero recomendado)
    torch.cuda.empty_cache()

def process_batch(input_folder='input', output_folder='outputs', duration=5.0, seed=31337, steps=25, use_teacache=True, gpu_memory_preservation=6, mp4_crf=16):
    """
    Process a batch of images from the input folder with corresponding prompts
    """
    # Crear carpetas si no existen
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    
    # Limpiar archivos previos en la carpeta de salida
    clean_previous_outputs(output_folder)

    # Verificar archivo de prompts
    prompt_file = os.path.join(input_folder, 'prompt.txt')
    if not os.path.exists(prompt_file):
        error_msg = f"Error: Prompt file not found at {prompt_file}"
        print(error_msg)
        yield error_msg, "", None, ""
        return
    
    # Inicializar parámetros
    current_params = {
        'duration': duration,
        'seed': seed,
        'steps': steps,
        'use_teacache': use_teacache,
        'gpu_memory_preservation': gpu_memory_preservation,
        'mp4_crf': mp4_crf
    }
    
    # Leer parámetros de batch_params.txt si existe
    batch_params_file = os.path.join(input_folder, 'batch_params.txt')
    if os.path.exists(batch_params_file):
        try:
            with open(batch_params_file, 'r') as f:
                for line in f:
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        key = key.strip()
                        if key in current_params:
                            if isinstance(current_params[key], bool):
                                current_params[key] = value.lower() in ['true', 'yes', '1', 't', 'y']
                            elif isinstance(current_params[key], int):
                                current_params[key] = int(value)
                            elif isinstance(current_params[key], float):
                                current_params[key] = float(value)
        except Exception as e:
            print(f"Error reading batch_params.txt: {str(e)}")
    
    print("Using parameters:")
    for key, value in current_params.items():
        print(f"  - {key}: {value}")
    
    # Leer prompts
    with open(prompt_file, 'r') as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    
    # Encontrar imágenes
    image_files = []
    for ext in ['png', 'jpg', 'jpeg']:
        image_files.extend(glob.glob(os.path.join(input_folder, f'*.{ext}')))
    
    image_files.sort()
    
    if not image_files:
        error_msg = f"No image files found in {input_folder}"
        print(error_msg)
        yield error_msg, "", None, ""
        return
    
    print(f"Found {len(image_files)} images to process")
    print(f"Found {len(prompts)} prompts to use")
    
    # Parámetros fijos
    n_prompt = ""
    latent_window_size = 9
    cfg = 1.0
    gs = 10.0
    rs = 0.0
    
    # Procesar cada imagen
    results = []
    last_output = None
    last_progress_html = ""
    
    for i, image_file in enumerate(image_files):
        if i >= len(prompts):
            prompt = prompts[-1]
        else:
            prompt = prompts[i]
        
        # Generar una seed única para este job
        current_seed = generate_random_seed() if seed == -1 else seed
        # Si seed es -1, se generará una aleatoria para cada job
        
        progress_msg = f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_file)} (Seed: {current_seed})"
        yield progress_msg, progress_msg, None, last_progress_html
        
        try:
            input_image = np.array(Image.open(image_file))
            batch_id = os.path.basename(image_file).split('.')[0]
            
            # Nombre del archivo final para este item
            final_output_filename = os.path.join(output_folder, f'batch_{batch_id}_final.mp4')
            
            # Eliminar archivos temporales previos de este item
            for prev_file in glob.glob(os.path.join(output_folder, f'batch_{batch_id}_*.mp4')):
                if prev_file != final_output_filename:
                    try:
                        os.remove(prev_file)
                    except:
                        pass
            
            # Guardar la imagen de entrada
            input_image_np = resize_and_center_crop(input_image, target_width=640, target_height=640)
            Image.fromarray(input_image_np).save(os.path.join(output_folder, f'batch_{batch_id}_input.png'))
            
            # Llamar a worker con los parámetros actuales
            worker(input_image, prompt, n_prompt, current_seed, current_params['duration'], 
                  latent_window_size, current_params['steps'], cfg, gs, rs, 
                  current_params['gpu_memory_preservation'], current_params['use_teacache'], 
                  current_params['mp4_crf'], batch_id, output_folder)
            
            while True:
                flag, data = global_stream.output_queue.next()
                
                if flag == 'file':
                    last_output = data
                    results.append(last_output)
                    yield f"Generated: {last_output}", progress_msg, gr.Video(value=last_output), last_progress_html
                
                if flag == 'progress':
                    preview, desc, html = data
                    last_progress_html = html
                    yield progress_msg, desc, gr.Video(value=last_output) if last_output else None, html
                
                if flag == 'batch_progress':
                    message = data
                    yield progress_msg, message, gr.Video(value=last_output) if last_output else None, last_progress_html
                
                if flag == 'end':
                    break
        
        except Exception as e:
            error_msg = f"Error processing {image_file}: {str(e)}"
            print(error_msg)
            results.append(error_msg)
            yield error_msg, progress_msg, None, last_progress_html
    
    # Resumen final
    success_count = len([r for r in results if not isinstance(r, str) or not r.startswith("Error")])
    error_count = len(results) - success_count
    
    summary = f"Batch processing complete!\nSuccess: {success_count}\nErrors: {error_count}"
    if error_count > 0:
        summary += "\nError details:\n" + "\n".join([r for r in results if isinstance(r, str) and r.startswith("Error")])
    
    if last_output is not None:
        yield summary, "Batch processing completed!", gr.Video(value=last_output), ""
    else:
        yield summary, "Batch processing completed!", None, ""

# Configuración de la interfaz Gradio
quick_prompts = [
    'The girl dances gracefully, with clear movements, full of charm.',
    'A character doing some simple body movements.',
]
quick_prompts = [[x] for x in quick_prompts]

#css = make_progress_bar_css()
css = """
.random-seed-btn {
    min-width: 60px !important;
    min-height: 50px !important;
    font-size: 30px !important;
    padding: 15px !important;
    margin-left: 10px !important;
}
""" + make_progress_bar_css()
block = gr.Blocks(css=css).queue()
with block:
    gr.Markdown('# FramePack-F1')
    
    with gr.Tabs():
        # Pestaña para generación individual
        with gr.TabItem("Single Image Generation"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(sources='upload', type="numpy", label="Image", height=320)
                    prompt = gr.Textbox(label="Prompt", value='')
                    example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Quick List', samples_per_page=1000, components=[prompt])
                    example_quick_prompts.click(lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt, show_progress=False, queue=False)

                    with gr.Row():
                        start_button = gr.Button(value="Start Generation", variant="primary")
                        end_button = gr.Button(value="End Generation", interactive=False)

                    gr.Markdown("### Generation Settings")
                    with gr.Group():
                        use_teacache = gr.Checkbox(label='Use TeaCache', value=True, info='Faster speed, but often makes hands and fingers slightly worse.')
                        
                        # Seed con botón random
                        with gr.Row():
                            seed = gr.Number(label="Seed", value=31337, precision=0)
                            random_seed_btn = gr.Button("🎲", 
                              variant="secondary", 
                              size="lg",  
                              elem_classes="random-seed-btn")
                        
                        total_second_length = gr.Slider(label="Total Video Length (Seconds)", minimum=1, maximum=120, value=5, step=0.1)
                        steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1, info='Changing this value is not recommended.')
                        gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01, info='Changing this value is not recommended.')
                        gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB)", minimum=6, maximum=128, value=6, step=0.1, info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed.")
                        mp4_crf = gr.Slider(label="MP4 Compression", minimum=0, maximum=100, value=16, step=1, info="Lower means better quality. 0 is uncompressed. Change to 16 if you get black outputs.")
                        
                        # Parámetros ocultos
                        n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)
                        latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False)
                        cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)
                        rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)

                with gr.Column(scale=1.2):
                    preview_image = gr.Image(label="Next Latents", height=200, visible=False)
                    result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=False, height=512, loop=True)
                    progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
                    progress_bar = gr.HTML('', elem_classes='no-generating-animation')
        
        # Pestaña para procesamiento por lotes
        with gr.TabItem("Batch Processing"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Batch Processing Settings")
                    batch_folder = gr.Textbox(label="Batch Input Folder", value="input", info="Folder containing images and prompt file")
                    batch_output_folder = gr.Textbox(label="Batch Output Folder", value="outputs", info="Folder to save generated videos")
                    
                    with gr.Row():
                        batch_duration = gr.Slider(label="Duration (seconds)", minimum=1, maximum=120, value=5, step=0.1)
                        
                    # Seed con botón random para batch
                    with gr.Row():
                        batch_seed = gr.Number(
                            label="Seed (-1 for random per job)", 
                            value=31337, 
                            precision=0,
                            info="Set to -1 to use a different random seed for each job"
                        )
                        batch_random_seed_btn = gr.Button("🎲", 
                            variant="secondary", 
                            size="lg", 
                            elem_classes="random-seed-btn"
                    )
                    with gr.Row():    
                        batch_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1)
                        batch_teacache = gr.Checkbox(label='Use TeaCache', value=True)
                    
                    with gr.Row():
                        batch_gpu_mem = gr.Slider(label="GPU Memory (GB)", minimum=6, maximum=128, value=6, step=0.1)
                        batch_crf = gr.Slider(label="MP4 Quality", minimum=0, maximum=100, value=16, step=1)

                    
                    batch_button = gr.Button(value="Start Batch Processing", variant="primary")
                    batch_status = gr.Markdown('')
                    
                    gr.Markdown("""
                    ### How to use Batch Processing:
                    1. Create a folder with your input images (.png, .jpg, .jpeg)
                    2. Add a file named 'prompt' with one prompt per line
                    3. Optional: Create 'batch_params.txt' to override settings
                    """)

                with gr.Column():
                    gr.Markdown("### Batch Processing Output")
                    batch_progress = gr.Textbox(label="Processing Status", interactive=False, lines=1)
                    batch_output = gr.Video(label="Latest Generated Video", autoplay=False, height=480)
                    batch_progress_bar = gr.HTML('', elem_classes='no-generating-animation')  # Nueva barra de progreso

    gr.HTML('<div style="text-align:center; margin-top:20px;">Share your results and find ideas at the <a href="https://x.com/search?q=framepack&f=live" target="_blank">FramePack Twitter (X) thread</a></div>')

    # Conexiones para generación individual
    single_ips = [input_image, prompt, n_prompt, seed, total_second_length, 
                 latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, 
                 use_teacache, mp4_crf]
    
    start_button.click(
        fn=process, 
        inputs=single_ips, 
        outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button]
    )
    end_button.click(
        fn=end_process,
        outputs=None,
        queue=False,  # ¡Importante! Evita bloquear la cola de Gradio
        preprocess=False
    )
    
    # Conexiones para procesamiento por lotes
    batch_button.click(
        fn=process_batch,
        inputs=[
            batch_folder,
            batch_output_folder,
            batch_duration,
            batch_seed,
            batch_steps,
            batch_teacache,
            batch_gpu_mem,
            batch_crf
        ],
        outputs=[batch_status, batch_progress, batch_output, batch_progress_bar]  # Añadido batch_progress_bar
    )
    
    # Conexiones para los botones random seed
    random_seed_btn.click(
        fn=generate_random_seed,
        outputs=[seed]
    )
    
    batch_random_seed_btn.click(
        fn=generate_random_seed,
        outputs=[batch_seed]
    )

# Ejecución
if args.batch:
    print("Running in batch mode...")
    process_batch(args.input_folder, args.output_folder, args.duration, args.seed, args.steps)
else:
    block.launch(
        server_name=args.server,
        server_port=args.port,
        share=args.share,
        inbrowser=args.inbrowser,
    )