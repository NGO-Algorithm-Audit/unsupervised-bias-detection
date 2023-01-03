# Unsupervised bias scan tool
## A quantitative method to inform qualitative bias testing 

☁️ The bias scan tool is available as a web application: https://www.algorithmaudit.eu/bias_scan/. 

📊 Main presentation: [slides](https://github.com/NGO-Algorithm-Audit/Bias_scan/blob/master/Main_presentation_bias_scan.pdf).

📄 Technical documentation: [report](https://github.com/NGO-Algorithm-Audit/Bias_scan/blob/master/Bias_scan_tool_report.pdf).

## Key takeaways – Why this approach?
- **Bias scan tool**: Quantitative method to _detect_ higher- and lower-dimensional forms of algorithmic differentiation;
- **Unsupervised bias detection**: No user data needed on protected attributes; 
- **Model-agnostic**: Works for all binary AI classifiers; 
- **Audit commission**: Qualitative, expert-led approach to _establish_ unfair treatment;
- **Open-source and not-for-profit**: Available for the entire AI Audit community.

## Solution overview
![image](./images/Quantitative-qualitative.png)
<sub>*Figure 1 – Quantitative-qualitative solution overview*</sub>

## Executive summary
Artificial intelligence (AI) is increasingly used to automate or support policy decisions that affects individuals and groups. It is imperative that AI adheres to the legal and ethical requirements that apply to such policy decisions. In particular, policy decisions should not be systematically discriminatory (direct or indirect) with respect to protected attributes such as gender, sex, ethnicity or race.

To achieve this, we propose a scalable, model-agnostic, and open-source bias scan tool to identify potentially discriminated groups of similar users in binary AI classifiers. This bias scan tool does not require *a priori* information about existing disparities and protected attributes, and is therefore able to detect possible proxy discrimination, intersectional discrimination and other types of differentiation that evade non-discrimination law. The tool is available as a web application, available on the [website](https://www.algorithmaudit.eu/bias_scan/) of Algorithm Audit, such that it can be used by a wide public.

As demonstrated on a BERT-based Twitter disinformation detection model, the bias scan tool identifies statistically significant disinformation classification bias against [...].

These observations do not establish prohibited *prima facie* discrimination. Rather, the identified disparities serve as a starting point to assess potential discrimination according to the context-sensitive legal doctrine, i.e., assessment of the legitimacy of the aim pursued and whether the means of achieving that aim are appropriate and necessary. For this qualitative assessment, we propose an expert-oriented deliberative method. Which allows policy makers, journalist, data subjects and other stakeholders to publicly review identified quantitative disparities against the requirements of non-discrimination law and ethics. In our two-pronged quantitative-qualitative solution, scalable statistical methods work in tandem with the normative capabilities of human subject matter experts to define fair AI on a case-by-case basis (see the solution overview Figure below). 

<sub>**Note**: The implemented bias scan tool is based on the k-means Hierarchical Bias-Aware Clustering (HBAC) method as described in Misztal-Radecka, Indurkya, *Information Processing and Management*. Bias-Aware Hierarchical Clustering for detecting the discriminated groups of users in recommendation systems (2021). Additional research indicates that k-means HBAC, in comparison to other clustering algorithms, works best to detect bias in real-world datasets.</sub>

## Input data bias scan tool
A .csv file of max. 1GB, with columns structured as follows: features, predicted labels, truth labels. Only the order, not the naming, of the columns is important.

- **Features**: unscaled numeric values, e.g., *feat_1, feat_2, ..., feat_n;*
- **Predicted label**: 0 or 1;
- **Truth label**: 0 or 1.

| feat_1 | feat_2 | ... | feat_n | pred_label | truth_label |
|--------|--------|-----|--------|------------|-------------|
| 10     | 1      | ... | 0.1    | 1          | 1           |
| 20     | 2      | ... | 0.2    | 1          | 0           |
| 30     | 3      | ... | 0.3    | 0          | 0           |

<sub>*Table 1 – Structure of input data in bias scan web application*</sub>

☁️ The tool is available as a web application on the [website](https://www.algorithmaudit.eu/bias_scan/) of Algorithm Audit.

## Case study – BERT-based disinformation classifier (Twitter1516 data set)
We use the bias scan tool to assess fair treatment of a self-trained disinformation detection algorithm on the Twitter1516 dataset. Below, statistically significant disparities found by the tool are presented. Based on these quantitative results, questions are distilled and asked to an audit commission consiting of AI experts. This audit commission formulates a normative advice if, and how, (higher-dimensional) unfair treatment can be assessed on the basis of the bias scan results.

### Bias scan pipeline
A BERT-based disinformation classifier is trained on true and false tweets (n=1,057) from the [Twitter1516](https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=0&file_subpath=%2Frumor_detection_acl2017) dataset. For this dataset, user and content features are [collected](https://github.com/NGO-Algorithm-Audit/Bias_scan/blob/master/data/Twitter_dataset/Twitter_API_data_collection.ipynb) from the Twitter API. More details on the training process of the BERT-based disinformation classifier can be found [here](https://github.com/NGO-Algorithm-Audit/Bias_scan/blob/master/classifiers/BERT_disinformation_classifier/BERT_Twitter_classifier.ipynb). 

![image](./images/HBAC_pipeline.png)
<sub>*Figure 2 – Bias scan pipeline for the disinformation classifier case study*</sub>

### Results: False Positive Rate (FPR) bias metric
For this bias scan, bias is defined by the False Positive Rate (FPR) per cluster. That is: 

_Bias = FPR(cluster) - FPR(rest of dataset)_. 

A False Positive (FP) means that true content is classified as disinformation by the AI classifier. The cluster with highest bias deviates 0.08 from the rest of the data set. There are 249 tweets in this cluster.
![image](./images/FPR_metric.png)
<sub>*Figure 3 – Bias scan results for FPR bias scan. Features with dark blue confidence intervals that do not hit the x=0 line indicate statistically significant difference in feature means between the cluster with highest bias and the rest of the dataset.*</sub>

On average, users that:
- are verified, have higher #followers, user engagement and #URLs;
- use less #hashags and have lower tweet length

have more true content classified as false (false positives).

<!-- This is the full list of statistical significant differences in (feature) means between the cluster with most bias (0.08) and rest of dataset:
| feature          | difference | p-value |
|------------------|------------|---------|
| verified         | 1.419      | 0.000   |
| #followers       | 0.777      | 0.000   |
| user_engagement  | 0.878      | 0.000   |
| #URLs            | 1.130      | 0.000   |
| #mentions        | -0.193     | 0.064   |
| #hashs           | -0.634     | 0.000   |
| length           | -0.669     | 0.000   |
| #sentiment_score | 0.167      | 0.115   |

*Table 2 – Bias scan results for FPR scan. A p-values below 0.05 indicates statistically significant difference in feature means between the cluster with highest bias and the rest of the dataset.* -->

More details on this case study can be found [here](https://github.com/NGO-Algorithm-Audit/Bias_scan/blob/master/HBAC_scan/HBAC_BERT_disinformation_classifier.ipynb). 

### Results: False Negative Rate (FNR) bias metric
For this bias scan, bias is defined by the False Negative Rate (FNR) per cluster. That is: 

_Bias = FNR(cluster) - FNR(rest of dataset)_. 

A False Negative (FN) means that disinformation is classified as true by the AI classifier. The cluster with highest bias deviates 0.13 from the rest of the data set. There are 46 tweets in this cluster.
![image](./images/FNR_metric.png)
<sub>*Figure 4 – Bias scan results for FNR bias scan. Features with dark blue confidence intervals that do not hit the x=0 line indicate statistically significant difference in feature means between the cluster with highest bias and the rest of the dataset.*</sub>

On average, users that:
- use more #hashtags and have higher sentiment score;
- are non-verified, have less #followers, user engagement and tweet length

have more false content classified as true (false negatives).

<!-- This is the full list of statistical significant differences in (feature) means between the cluster with most bias (0.13) and rest of dataset:
| feature          | difference | p-value |
|------------------|------------|---------|
| verified         | -1.965     | 0.000   |
| #followers       | -0.575     | 0.000   |
| user_engagement  | -0.619     | 0.000   |
| #URLs            | -0.080     | 0.607   |
| #mentions        | -0.086     | 0.465   |
| #hashs           | 0.588      | 0.005   |
| length           | -0.702     | 0.000   |
| #sentiment_score | 0.917      | 0.000   |

*Table 3 – Bias scan results for FNR scan. A p-values below 0.05 indicates statistically significant difference in feature means between the cluster with highest bias and the rest of the dataset.* -->

More details on this case study can be found [here](https://github.com/NGO-Algorithm-Audit/Bias_scan/blob/master/HBAC_scan/HBAC_BERT_disinformation_classifier.ipynb).  

### Audit commission: Qualitative assessment of identified disparities
These above quantitative disparities do not establish prohibited _prima facie_ discrimination. Rather, the identified disparities serve as a starting point to assess potential unfair treatment according to the context-sensitive qualitative doctrine. To assess unfair treatment, we question a commission of experts:
1. Is there an indication that one of the statistically significant features, or a combination of the features, stated in Figure 2-3 are critically linked to one or multiple protected grounds? 
2. In the context of disinformation detection, is it as harmful to classify true content as false (false positive) as false content as true (false negative)?
3. For a specific cluster of people, is it justifiable to have true content classified as false 8 percentage points more often? For a specific cluster of people, is it justifiable to have false content classified as true 13 percentage points more often?
4. Considering the disparate treatment of users with a verified profile, above average sentiment score and/or below average number of URLs used in their tweets, could the observed disparate treatment be perceived as ethically undesirable?

## Conclusion
The audit commissions convenes in Jan-Feb 2023, to elaborate on the above questions.

## Structure of this repository
```
    .
    ├── HBAC_scan                                               # Unsupervised bias scan (quantitative)
    ├── audit_commission                                        # Audit commission materials (qualitative)
    ├── classifiers                                             # Self-trained binary AI classifiers
    ├── data                                                    # Twitter1516 and German Credit data
    ├── images                                                  # Images
    ├── literature                                              # Reference materials
    ├── .gitattributes                                          # To store large files
    ├── .gitignore                                              # Files to be ignored in this repo
    ├── LICENSE                                                 # MIT license for sharing 
    ├── Main_presentation_bias_scan.pdf                         # Main presentation (slides)
    ├── README.md                                               # Readme file 
    └── Technical_documentation_bias_scan.docx                  # Techical documentation (report)
```
## Contributors and endorsements
### Algorithm Audit's bias scan tool team:
- Jurriaan Parie, Trustworthy AI consultant at Deloitte
- Ariën Voogt, PhD-candidate in Philosophy at Protestant Theological University of Amsterdam
- Joel Persson, PhD-candidate in Applied Data Science at ETH Zürich

### This project is endorsed by:
#### Journalism
- Gabriel Geiger, Investigative Reporter Algorithms and Automated Decision-Making at Lighthouse Reports
- AA
- BB

#### Industry
- Laurens van der Maas, Data Scientist at AWS;
- CC
- DD

#### Academia
- Anne Meuwese, Professor in Public Law & AI at Leiden University
- Hinda Haned [to be confirmed], Professor in Data Science at University of Amsterdam
- Marlies van Eck, Assistant Professor in Administrative Law & AI at Radboud University
- Emma Beauxis-Ausselet [to be confirmed], Associate Professor Ethical Computing at University of Amsterdam
- Vahid Niamadpour, PhD-candidate in Linguistics at Leiden University
- Floris Holstege, PhD-candidate in Explainable Machine Learning at University of Amsterdam

#### Civil society organisations
- XX
- YY
- Simone Maria Parazzoli, Intern at the OECD Observatory of Public Sector Innovation (OPSI).