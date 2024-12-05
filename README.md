# Measuring Bias in Contextual Embeddings Across Arabic and Western Terms
_Lujain Khalil, Dara Varam, Arwa Bayoumy and Dr. Alex Aklson_

This repository contains the necessary code to run and replicate results for our paper, titled "Dates Over Donuts: Measuring Embeddings Biases Across Arabic and Western Terms."

We also include the results included in the paper, as generated through the code. A small tutorial on running experiments is also included for the viewer's convenience. 

![alt text](https://github.com/lujain-khalil/MLR503-Project/blob/main/Figures/MLR503%20System%20Overview.png)

## Introduction

## How to use our code:
1. To extract embeddings from a specific model, run ```python src/extract_embeddings.py model_name```
2. To run all evaluation scripts for a specific model, run ```python main.py model_name```
3. To run a specific evaluation script (```eval_clustering.py```, ```eval_norms.py```, or ```eval_association.py```), run ```python src/eval_clustering.py model_name```
4. To compare association scores across models, run ```python src/model_comparision.py```

Note: The models that are currently supported are as follows:

```python
SUPPORTED_MODELS = {
    "xlm-roberta-base": "xlm-roberta-base",
    "mbert": "bert-base-multilingual-cased",
    "distilbert": "distilbert/distilbert-base-multilingual-cased",
    "bert":"google-bert/bert-base-multilingual-uncased",
    "xlm-roberta-large": "xlm-roberta-large",

    "arabert": "aubmindlab/bert-base-arabertv2",  
    "arabertlarge":"aubmindlab/bert-large-arabertv02",
    "marbert": "UBC-NLP/MARBERTv2",
    "arbert": "UBC-NLP/ARBERTv2", 
    "camelbert": "CAMeL-Lab/bert-base-arabic-camelbert-mix",
}
```
To extend the framework to more models, change the ```SUPPORTED_MODELS``` variable found in ```src/utils.py```. You are encouraged to do this if you would like to extend the framework beyond what is already being presented in the study. 

## Sample of results

### Association Results for MARBERT
| ![Image 1]([path/to/image1.png](https://github.com/lujain-khalil/MLR503-Project/blob/main/Figures/MARBERT/Association/MARBERT%20same_sentences.png)) | ![Image 2]([path/to/image2.png](https://github.com/lujain-khalil/MLR503-Project/blob/main/Figures/MARBERT/Association/MARBERT%20same_terms.png)) | ![Image 3]([path/to/image3.png](https://github.com/lujain-khalil/MLR503-Project/blob/main/Figures/MARBERT/Association/MARBERT%20seat_association_scores.png)) |
|:------------------------------:|:------------------------------:|:------------------------------:|
| SAME score for sentences                     | SAME score for terms                     | SEAT association score                     |

| ![Image 4]([path/to/image4.png](https://github.com/lujain-khalil/MLR503-Project/blob/main/Figures/MARBERT/Association/MARBERT%20seat_scores.png)) | ![Image 5]([path/to/image5.png](https://github.com/lujain-khalil/MLR503-Project/blob/main/Figures/MARBERT/Association/MARBERT%20weat_association_scores.png)) | ![Image 6]([path/to/image6.png](https://github.com/lujain-khalil/MLR503-Project/blob/main/Figures/MARBERT/Association/MARBERT%20weat_scores.png)) |
|:------------------------------:|:------------------------------:|:------------------------------:|
| SEAT scores                   | WEAT association scores                    | WEAT scores                    |




## Citation and reaching out
If you have found our work useful for your own research, we encourage you to cite us with the below (the paper has not been published yet, we will update this as necessary): 

- ### BibTeX:


```
@Article{dates-over-donuts,
AUTHOR = {Khalil, Lujain and Varam, Dara and Bayoumy, Arwa and Aklson, Alex},
TITLE = {Dates Over Donuts: Measuring Embeddings Biases Across Arabic and Western Terms},
JOURNAL = { },
VOLUME = {},
YEAR = {},
NUMBER = {},
ARTICLE-NUMBER = {},
URL = {},
ISSN = {},
ABSTRACT = {},
DOI = {}
}
```

You are also welcome to reach out through email (g00082632@alumni.aus.edu - Lujain Khalil or b00081313@alumni.aus.edu - Dara Varam)
