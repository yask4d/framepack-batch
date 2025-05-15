# FramePack-F1 (Modificación para Batch Processing)  
# Video Generation Tool  

## Descripción  

Esta versión es una modificación del proyecto original **[FramePack de lllyasviel](https://github.com/lllyasviel/FramePack)** que añade funcionalidad de procesamiento por lotes (batch) y otras mejoras.  

FramePack-F1 es una herramienta avanzada para generar videos a partir de imágenes estáticas utilizando inteligencia artificial. El sistema utiliza modelos de difusión para crear animaciones fluidas basadas en una imagen de entrada y un prompt descriptivo.  

**Mejoras principales respecto al original:**  
- Soporte completo para procesamiento por lotes (batch)  
- Interfaz gráfica mejorada con Gradio  
- Optimización de memoria para diferentes configuraciones de GPU  
- Configuración simplificada para usuarios  

## Características principales  

- Generación de videos a partir de imágenes estáticas  
- **Nuevo:** Procesamiento por lotes automático  
- Soporte para procesamiento individual y batch  
- Interfaz gráfica intuitiva con Gradio  
- Control preciso sobre parámetros de generación  
- Optimización de memoria para GPUs con diferentes capacidades  

## Instalación simplificada  

1. Primero instala el FramePack original siguiendo las instrucciones de:  
   https://github.com/lllyasviel/FramePack  

2. Luego descarga estos dos archivos de nuestra modificación:  
   - [run-batch.bat](enlace_al_archivo) - Colócalo en la raíz de FramePack  
   - [batch.py](enlace_al_archivo) - Colócalo en la carpeta `webui` de FramePack  

3. Ejecuta `run-batch.bat` para iniciar la interfaz con las mejoras  

## Instalación completa (opcional)  

Si prefieres clonar todo el repositorio:  

1. Clona el repositorio:  
   ```bash  
   git clone [URL del repositorio]  
   cd [nombre del repositorio]  
   ```  

2. Instala las dependencias:  
   ```bash  
   pip install -r requirements.txt  
   ```  

3. Configura las variables de entorno (opcional):  
   ```bash  
   export HF_HOME=./hf_download  
   ```  

## Uso  

### Interfaz gráfica mejorada  

Ejecuta:  
```bash  
python batch.py  
```  

Opciones disponibles:  
- `--share`: Habilita el acceso remoto  
- `--server`: Especifica la dirección IP del servidor  
- `--port`: Especifica el puerto  
- `--inbrowser`: Abre automáticamente en el navegador  

### Modo batch (nueva funcionalidad)  

Para procesamiento por lotes:  
```bash  
python batch.py --batch --input_folder [carpeta_entrada] --output_folder [carpeta_salida] --duration [duración] --seed [semilla] --steps [pasos]  
```  

### Estructura para batch processing  

1. Crea una carpeta con las imágenes de entrada (.png, .jpg, .jpeg)  
2. Añade un archivo llamado 'prompt' con un prompt por línea  
3. Opcional: Crea 'batch_params.txt' para sobreescribir configuraciones  

## Parámetros importantes  

- **Seed**: Semilla para la generación aleatoria  
- **Total Video Length**: Duración del video en segundos  
- **Steps**: Número de pasos de inferencia  
- **Distilled CFG Scale**: Controla la adherencia al prompt  
- **GPU Memory Preservation**: Memoria a reservar para otros procesos  
- **MP4 Compression**: Calidad del video de salida (0=mejor calidad)  

## Ejemplos de prompts  

- "The girl dances gracefully, with clear movements, full of charm."  
- "A character doing some simple body movements."  

## Soporte y comunidad  

Comparte tus resultados y encuentra ideas en el [hilo de FramePack en Twitter (X)](https://x.com/search?q=framepack&f=live)  

## Notas  

- Para GPUs con menos de 60GB de VRAM, el modo High-VRAM se desactiva automáticamente  
- La calidad puede variar según los parámetros utilizados  
- El procesamiento batch puede ser intensivo en recursos para muchas imágenes  

## Agradecimientos  

Este proyecto está basado en el trabajo original de lllyasviel:  
https://github.com/lllyasviel/FramePack  

## Licencia  

Apache-2.0 license  

# Cite  

    @article{zhang2025framepack,  
        title={Packing Input Frame Contexts in Next-Frame Prediction Models for Video Generation},  
        author={Lvmin Zhang and Maneesh Agrawala},  
        journal={Arxiv},  
        year={2025}  
    }
