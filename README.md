# Image Captioning — CNN + LSTM/GRU

Generación automática de descripciones de imágenes con arquitectura
Encoder-Decoder (ResNet-50 + LSTM/GRU) entrenada sobre MS-COCO 2014.

**BLEU-4 = 0.2933** — supera el baseline de Vinyals et al. (2015)

## Estructura del proyecto

\`\`\`
image_captioning/
├── checkpoints/          ← COLOCA AQUI LOS MODELOS (ver abajo)
│   ├── best_model.ckpt   ← modelo principal (LSTM fine-tuned)
│   └── base_model.ckpt   ← modelo base (encoder congelado)
├── data/
│   ├── vocab.pkl         ← vocabulario pre-construido (incluido)
│   └── coco/             ← dataset COCO (descargar con script)
├── imagenes_validation/  ← 12 imágenes de validación externa
├── results/              ← curvas de entrenamiento y BLEU
├── models/               ← encoder.py  decoder.py
├── config.py
├── train.py
├── evaluate.py
├── validate_external.py
├── api.py
└── requirements.txt
\`\`\`

## Modelos preentrenados

Los archivos .ckpt superan el límite de 100 MB de GitHub y no se
incluyen en el repositorio. Cópialos manualmente a checkpoints/:

| Archivo | Tamaño | Descripción |
|---------|--------|-------------|
| best_model.ckpt | ~384 MB | LSTM fine-tuned, beam k=3 — usar para la API |
| base_model.ckpt | ~201 MB | LSTM encoder congelado — para comparativa BLEU |

## Instalación

\`\`\`bash
# PyTorch CPU (laptop sin GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# PyTorch GPU (si hay CUDA disponible)
pip install torch torchvision

# Resto de dependencias
pip install -r requirements.txt
\`\`\`

## API en tiempo real

\`\`\`bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
\`\`\`

Abrir http://localhost:8000 en el navegador.
Solo requiere best_model.ckpt y data/vocab.pkl — no necesita COCO.

## Entrenamiento desde cero

\`\`\`bash
./download_coco.sh                               # MS-COCO 2014 (~19 GB)
python train.py                                  # LSTM con fine-tuning
python train.py --cell_type gru --tag gru_variant
\`\`\`

## Evaluación BLEU

\`\`\`bash
python evaluate.py     --ckpt      checkpoints/best_model.ckpt     --base_ckpt checkpoints/base_model.ckpt
\`\`\`

## Validación externa

\`\`\`bash
python validate_external.py     --base    checkpoints/base_model.ckpt     --variant checkpoints/best_model.ckpt
\`\`\`

## Autores

Jonatan D. Navarrete Gómez
Universidad de Guanajuato 
