# Automatically Generating Counterfactually Augmented Data (CAD)

This repository contains materials automatically generate CAD using Polyjuice, ChatGPT, and Flan-T5 and compare these against manual CAD wrt efficacy as training data, particularly for increasing out-of-domain generaizability of sexism and hate speech detection models. Detailed information can be found in our paper:

*Sen, I., Assenmacher, D., Samory, M., Augenstein, I., Aalst, W.V., & Wagner, C. (2023). [People Make Better Edits: Measuring the Efficacy of LLM-Generated Counterfactually Augmented Data for Harmful Language Detection.](https://arxiv.org/abs/2311.01270) ArXiv, abs/2311.01270. To Appear at EMNLP'23*

### Data

We make the automatically generated counterfactuals available in 'data' folder. Due to licensing restrictions, we cannot include the text of the other datasets in this repository. Therefore, the text columns to be labeled in the in-domain and out-of-domain datasets are not included in the data files. However, we include:
- the links to where they can be downloaded from (at the end of this page)
- the labels given to these items by our classifiers and ChatGPT. These can be found in the 'results/intermediate/' folder.

### Code structure

[1] CAD generation

Polyjuice: generate_polyjuice_cad.py
ChatGPT: chatgpt CAD generation.ipynb
FlanT5: FLAN-T5.ipynb

[2] Combine all automated CADs with manual CADs: training data prepping.ipynb which will create the paired files (automated CADs included in the 'data' folder)

[3] Model training

Sexism: sexism run models.ipynb
Hate speech: hatespeech run models.ipynb

Running these scripts will also use the trained models to predict labels for the test sets which are saved to 'results/intermediate/'

[4] Baselines
ChatGPT: chatgpt fewshot labeling.ipynb
Flan T5: FLAN-T5.ipynb
Perspective API: augment_data_perspective.py

[5] RQ1 results: rq1 collate results.ipynb (Fig 1, 4) which also includes the preprocessing of Flan T5 and ChatGPT few-shot labeling outputs as well as how much of their outputs do not conform to our expected labels (Table 11). 

[6] Descriptive properties: rq2 CAD properties.ipynb (Fig 5-8 in the appendix)
This will create the csv files with properties of the different types of CADs required for the regression analysis and save them in results/intermediate/

[7] V-Information based analysis:  rq2 which cad v_info.ipynb contains the following:
Generate train-test splits for getting the datasets used for getting the V-info scores
Descriptive analysis (Fig 3, Table 8, 9)
Regression analysis (Fig 2, Table 5, 10)

[8] To run the V-information analysis:
Download the code from https://github.com/kawine/dataset_difficulty 
Obtain the train-test data using 7a
Run the code in autocad_vinfo_commands.txt


### LINKS TO EXTERNAL DATASETS

#### sexism

ID	https://search.gesis.org/research_data/SDN-10.7802-2251

OOD1	https://github.com/fhstp/EXIST2022/tree/main/data/EXIST2022_orig

OOD2	https://github.com/ellamguest/online-misogyny-eacl2021/tree/main/data

OOD3	https://github.com/rewire-online/edos

Hatecheck (HC)	https://github.com/paul-rottger/hatecheck-data

#### hate speech

ID	https://github.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset

OOD1	https://github.com/jing-qian/A-Benchmark-Dataset-for-Learning-to-Intervene-in-Online-Hate-Speech/blob/master/data/gab.csv

OOD2	http://hatespeech.di.unito.it/hateval.html

OOD3	https://github.com/jing-qian/A-Benchmark-Dataset-for-Learning-to-Intervene-in-Online-Hate-Speech/blob/master/data/reddit.csv

OOD4	https://hasocfire.github.io/hasoc/2020/dataset.html

Hatecheck (HC)	https://github.com/paul-rottger/hatecheck-data
