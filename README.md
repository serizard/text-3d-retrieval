# text-3d-retrieval with OpenShape
---
![figure](https://github.com/user-attachments/assets/9fa9cd58-ef7c-4cc9-9c69-cdb0ed2bdbb1)



Research project at **_AI·Robotics Institute, KIST_**

This project aims to retrieve shapes from ModelNet40 based on user text input and visualizes them. Users can input text in various languages, and the number of retrieved shapes (k) can be specified arbitrarily.


## Process
Once an user inputs the text, it is refined by a GPT-2 model fine-tuned with a few-shot learning (20-shot) approach. This refined text is then encoded by the CLIP model, producing cosine similarity vector through dot products with the embeddings of ModelNet40. Depending on the settings, reranking is performed within k * 3 candidates based on the similarity between the features of keyword (in model name) and the user input features, resulting in more accurate retrieval outcomes. (Optional)


  


| Model                    | Name                 |
|--------------------------|----------------------|
| Prompt Engineering Model | [GPT-2](https://huggingface.co/openai-community/gpt2)           |
| Text Encoder             | [OpenCLIP ViT-bigG-14](https://github.com/mlfoundations/open_clip) |
| Shape Encoder            | [PointBERT](https://github.com/Colin97/OpenShape_code)            |




## Download dataset
```
python prepare_dataset.py --download-dir {directory path to download dataset}
```



## Inference
```
python main.py
```



## Demo
Demo is available in ```demo.ipynb```. It receives an user input and retrieve k corresponding shapes.

