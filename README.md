# Model Sizer

A library for you to check the size of models hosted on the 🤗 Hub and see the 
minimum recommended vRAM to load a model in with 🤗 Accelerate.

## Install Directions

`pip install git+https://github.com/muellerzr/model-sizer`

## Usage

From the CLI, you can check any model name on the 🤗 Hub so long as it has an
integration with 🤗 Transformers or `timm` (🤗 `diffusers` coming soon).

```bash
sizeup --model_name "togethercomputer/LLaMA-2-7B-32K" --dtypes float32 float16
```
```bash
Loading pretrained config for `togethercomputer/LLaMA-2-7B-32K` from `transformers`...
+----------------------------------------------------+
| Memory Usage for `togethercomputer/LLaMA-2-7B-32K` |
+------------+---------------------+-----------------+
|   dtype    |    Largest Layer    |    Total Size   |
+------------+---------------------+-----------------+
|  float32   |       500.0 MB      |     25.61 GB    |
|  float16   |       250.0 MB      |     12.81 GB    |
+------------+---------------------+-----------------+
```

```bash
sizeup --model_name "timm/resnet50.a1_in1k" --dtypes float32 float16
```
```bash
Loading pretrained config for `timm/resnet50.a1_in1k` from `timm`...
+------------------------------------------+
| Memory Usage for `timm/resnet50.a1_in1k` |
+---------+----------------+---------------+
|  dtype  | Largest Layer  |   Total Size  |
+---------+----------------+---------------+
| float32 |     9.0 MB     |    97.7 MB    |
| float16 |     4.5 MB     |    48.85 MB   |
+---------+----------------+---------------+
```