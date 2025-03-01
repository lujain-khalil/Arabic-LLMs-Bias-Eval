# Exploring Cultural Biases of Arabic-Speaking Large Language Models: A Word Embedding Perspective

_Lujain Khalil, Arwa Bayoumy, Dara Varam, and Dr. Alex Aklson_

This repository contains the necessary code to run and replicate results for our paper, titled "Dates Over Donuts: Measuring Embedding Biases Across Arabic and Western Terms."

We also include the paper results, as generated through the code. A small tutorial on running experiments is also included for the viewer's convenience.

![alt text](https://github.com/lujain-khalil/Arabic-LLMs-Bias-Eval/blob/main/Figures/L%20Khalil%20-%20System%20Overview.png)

## Introduction

## How to use our code:

1. To extract embeddings from a specific model, run ``python src/extract_embeddings.py model_name``
2. To run all evaluation scripts for a specific model, run ``python main.py model_name``
3. To run a specific evaluation script (``eval_clustering.py``, ``eval_norms.py``, or ``eval_association.py``), run ``python src/eval_clustering.py model_name``
4. To compare association scores across models, run ``python src/model_comparision.py``

Note: The models that are currently supported are as follows:

```python
SUPPORTED_MODELS = {
    # Multiligual
    "BERT":"google-bert/bert-base-multilingual-uncased",
    "mBERT": "bert-base-multilingual-cased",
    "DistilBERT": "distilbert/distilbert-base-multilingual-cased",
    "XLM-RoBERTa-Base": "xlm-roberta-base",
    "XLM-RoBERTa-Large": "xlm-roberta-large",

    # Monolingual
    "AraBERT": "aubmindlab/bert-base-arabertv2",  
    "AraBERT-Large":"aubmindlab/bert-large-arabertv02",
    "ARBERT": "UBC-NLP/ARBERTv2", 
    "CAMeLBERT": "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    "MARBERT": "UBC-NLP/MARBERTv2",
}
```

To extend the framework to more models, change the ``SUPPORTED_MODELS`` variable found in ``src/utils.py``. You are encouraged to do this if you would like to extend the framework beyond what is already being presented in the study.

## Sample of results

If you run the above code for any of the models, you will get the following figures generated once. In the below example, we will be working with ``marbert``. The code to do this would be ``python main.py marbert``

### Association Results for MARBERT

| ![Image 1](https://github.com/lujain-khalil/MLR503-Project/blob/main/Figures/MARBERT/Association/MARBERT%20same_sentences.png) | ![Image 2](https://github.com/lujain-khalil/MLR503-Project/blob/main/Figures/MARBERT/Association/MARBERT%20same_terms.png) | ![Image 3](https://github.com/lujain-khalil/MLR503-Project/blob/main/Figures/MARBERT/Association/MARBERT%20seat_association_scores.png) |
| :--------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------: |
|                                                   SAME score for sentences                                                   |                                                   SAME score for terms                                                   |                                                        SEAT association score                                                        |

| ![Image 4](https://github.com/lujain-khalil/MLR503-Project/blob/main/Figures/MARBERT/Association/MARBERT%20seat_scores.png) | ![Image 5](https://github.com/lujain-khalil/MLR503-Project/blob/main/Figures/MARBERT/Association/MARBERT%20weat_association_scores.png) | ![Image 6](https://github.com/lujain-khalil/MLR503-Project/blob/main/Figures/MARBERT/Association/MARBERT%20weat_scores.png) |
| :-----------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------: |
|                                                        SEAT scores                                                        |                                                        WEAT association scores                                                        |                                                        WEAT scores                                                        |

### Clustering Results for MARBERT

| ![Image 1](https://github.com/lujain-khalil/MLR503-Project/blob/main/Figures/MARBERT/Clustering/MARBERT%20kmeans_clusters.png) | ![Image 2](https://github.com/lujain-khalil/MLR503-Project/blob/main/Figures/MARBERT/Clustering/MARBERT%20tsne_plot_culture.png) |
| :--------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------: |
|                                               K-Means Clustering (2 clusters)                                               |                                                   tSNE plot (across culture)                                                   |

| ![Image 3](https://github.com/lujain-khalil/MLR503-Project/blob/main/Figures/MARBERT/Clustering/MARBERT%20tsne_plot_culture_entity.png) | ![Image 4](https://github.com/lujain-khalil/MLR503-Project/blob/main/Figures/MARBERT/Clustering/MARBERT%20tsne_plot_entity.png) |
| :-----------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------: |
|                                                   tSNE plot (across culture-entity)                                                   |                                                   tSNE plot (across entity)                                                   |

### Norms for MARBERT

| ![Image 1](https://github.com/lujain-khalil/MLR503-Project/blob/main/Figures/MARBERT/Norms/MARBERT%20boxplot_norms.png) | ![Image 2](https://github.com/lujain-khalil/MLR503-Project/blob/main/Figures/MARBERT/Norms/MARBERT%20cdf_norms.png) | ![Image 3](https://github.com/lujain-khalil/MLR503-Project/blob/main/Figures/MARBERT/Norms/MARBERT%20culture_entity_comparison.png) |
| :-------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------: |
|                                                   Box-plot of norms                                                   |                                                     CDF norms                                                     |                                                    Norms by culture and entity                                                    |

| ![Image 4](https://github.com/lujain-khalil/MLR503-Project/blob/main/Figures/MARBERT/Norms/MARBERT%20histogram_kde_norms.png) | ![Image 5](https://github.com/lujain-khalil/MLR503-Project/blob/main/Figures/MARBERT/Norms/MARBERT%20violin_plot_norms.png) |
| :-------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------: |
|                                                   Distribution histogram                                                   |                                                        Violin plot                                                        |

Keep in mind that running the above does not only generate these plots, but it also saves relevant metrics and information as ``.json`` files as well. The above is just a sample of what you can expect from running the code.

## Citation and reaching out

If you have found our work useful for your own research, we encourage you to cite us with the below (the paper has not been published yet, we will update this as necessary):

- ### BibTeX:

```
@Article{khalil2025culturalbiasllm,
AUTHOR = {Khalil, Lujain and Bayoumy, Arwa and Varam, Dara and Aklson, Alex},
TITLE = {Exploring Cultural Biases of Arabic-Speaking Large Language Models: A Word Embedding Perspective},
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

You are also welcome to reach out through email (g00082632@alumni.aus.edu - Lujain Khalil, g00082596@alumni.aus.edu - Arwa Bayoumy, or b00081313@alumni.aus.edu - Dara Varam)
