# text-3d-retrieval with OpenShape
---
![figure](https://github.com/user-attachments/assets/9fa9cd58-ef7c-4cc9-9c69-cdb0ed2bdbb1)



Research project at **_AI·Robotics Institute, KIST_**

This project retrieves shapes from ModelNet40 based on user text input and visualizes them. Users can input text in various languages, and the number of retrieved shapes (k) can be specified arbitrarily.

| Model                    | Name                 |
|--------------------------|----------------------|
| Prompt Engineering Model | [llama-3 8B] (https://huggingface.co/meta-llama/Meta-Llama-3-8B)           |
| Text Encoder             | [OpenCLIP ViT-bigG-14] (https://github.com/mlfoundations/open_clip) |
| Shape Encoder            | [PointBERT] (https://github.com/Colin97/OpenShape_code)            |


### Download dataset
'''
python prepare_dataset.py --download-dir {directory path to download dataset}
'''


### Inference
'''
python main.py --access-token {your huggingface access token}
'''

