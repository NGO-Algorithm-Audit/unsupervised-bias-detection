![image](./images/Header_Github.png)
## an expert-led, deliberative audit informed by a quantitative bias scan

☁️ The bias scan tool is available as a web application: https://www.algorithmaudit.eu/bias_scan/. 

⚖️ Algorithm Audit's case repository: https://www.algorithmaudit.eu/cases/.

📊 Main presentation: [slides](https://github.com/NGO-Algorithm-Audit/Bias_scan/blob/master/Main_presentation_bias_scan.pdf).

📄 Technical documentation: [report](https://github.com/NGO-Algorithm-Audit/Bias_scan/blob/master/Technical_documentation_bias_scan.pdf).

## Key takeaways – Why this approach?
- **Bias scan tool**: Quantitative method to _detect_ higher- and lower-dimensional forms of algorithmic differentiation;
- **Unsupervised bias detection**: No user data needed on protected attributes; 
- **Model-agnostic**: Works for all binary AI classifiers; 
- **Audit commission**: Qualitative, expert-led approach to _establish_ unfair treatment;
- **Case repository**: Over time a case repository emerges from which data scientists and public authorities can distill ‘techno-ethical’ best-practices;
- **Open-source and not-for-profit**: Available for the entire AI auditing community.

## Solution overview
Problem 1: The human mind is not equipped to detect higher-dimensional forms of algorithmic differentiation.

Problem 2: If differentiation is detected, a persistent gap remains between quantitative fairness metrics and qualitative interpretation
![image](./images/Quantitative-qualitative.png)
<sub>*Figure 1 – Quantitative-qualitative solution overview*</sub>

## Executive summary
Bias in machine learning models can have far-reaching and detrimental effects. With the increasing use of AI to automate or support decision-making in a wide variety of fields, there is a pressing need for bias assessment methods that take into account both the statistical detection and the social impact of bias in a context-sensitive way. This is why we present a comprehensive, two-pronged approach to addressing algorithmic bias. Our approach comprises two components: (1) a quantitative bias scan tool, and (2) a qualitative, deliberative assessment. In this way, the scalable and data-driven benefits of machine learning work in tandem with the normative and context-sensitive judgment of human experts, in order to determine fair AI in a concrete way.

Our bias scan tool, which forms the quantitative component, aims to discover complex and hidden forms of bias. The tool is specifically geared towards detecting unforeseen forms of bias and higher-dimensional forms of bias. Aside from unfair biases with respect to established protected groups, such as gender, sexual orientation, and race, bias can also occur with respect to other and unexpected groups of people. These forms of bias are more difficult to detect for humans, especially when the unfairly treated group is defined by a high-dimensional mixture of features. Our bias scan tool is based on an unsupervised clustering method, which makes it capable of detecting these complex forms of bias. It thereby tackles the difficult problem of detecting proxy-discrimination that stems from unforeseen and higher-dimensional forms of bias, including intersectional forms of discrimination. 

Informed by the quantitative results of our bias scan tool, subject-matter experts carry out an evaluative assessment in a deliberative way. The deliberative assessment provides a deeper understanding and evaluation of the detected bias. Such an assessment is essential for a balanced, normative assessment of the bias and its potential consequences in terms of social impact. This is because factual observation of quantitative discrepancies does not establish discrimination or unfair bias by itself. Instead, quantitative discrepancies can only serve as a starting point for evaluating the possibility of unfair treatment in a qualitative way, where the qualitative assessment takes into account the particular social context and the relevant legal doctrines. Human interpretation is thereby required to establish which statistical disparities do indeed qualify as unfair bias. The diversity of our commission of consulted experts contributes to a context-sensitive and multi-perspectival evaluation of the particular use case under investigation. This expert-led, deliberative approach is commonly used by NGO Algorithm Audit to provide ethical guidance on issues that arise in concrete algorithmic use cases.

Regarding the technical properties of our bias scan tool: we developed and implemented our tool as a scalable, user-friendly, free-to-use, and open-source web application. The tool works on all sorts of binary AI classifiers and is therefore model-agnostic. This design choice was intentional, as it enables users to audit the specific classifier applied in their use case, whatever it may be. The tool is based on the k-means Hierarchical Bias-Aware Clustering (HBAC) method as described in recent academic literature.  Using this method, the tool is able to identify groups of similar users (called ‘clusters’) that are potentially treated unfairly by a binary AI classifier. Importantly, our bias scan tool does not require a priori information about existing disparities and protected attributes. It thereby allows for detecting possible proxy discrimination, intersectional discrimination and other types of (higher-dimensional) unfair differentiation not easily conceived by humans. The tool identifies clusters of users that face a higher misclassification rate compared to the rest of the data set. The classification metric by which bias is defined can be manually chosen by the user in advance: false negative rate, false positive rate, or accuracy. The tool is available as a web application on the [website](https://www.algorithmaudit.eu/bias_scan/) of Algorithm Audit. It is thereby freely available for use by the AI auditing community.

To evaluate and demonstrate our approach for practice, we applied our two-pronged assessment method on a BERT-based disinformation detection model, which is trained on the widely cited Twitter1516 data set. For the quantitative part, our bias scan identifies a statistically significant bias in misclassification rate against a cluster of similar users (characterized by a verified profile, higher number of followers and higher user engagement score). On average, users belonging to this cluster face more true content being classified as false (false positives). We also find a bias with respect to another cluster of users, of whom false content is more often classified as true (false negatives). Our results are robust against alternative configurations of the classification algorithm. In particular, we re-perform our analysis with 162 different configurations of the hyperparameters of the classification algorithm, yielding over 1000 different clustering results. Aggregating these results produces the same findings as our main analysis.

On the basis of our results, we formulated a set of pressing questions about the disinformation classifier performance, and presented these questions to an independent audit commission composed of experts on AI and disinformation detection. Building on the quantitative results of the bias scan, these experts provided substantiated, normative judgments on whether the classifier is causing unfair treatment or not.

[INSERT RESULT OF DELIBERATIVE ROUND]

We visually present the results of our case study and our general approach to bias assessment in the [slide deck](https://github.com/NGO-Algorithm-Audit/Bias_scan/blob/master/Main_presentation_bias_scan.pdf). In-depth discussion of the technical details can be found in the [technical documentation](https://github.com/NGO-Algorithm-Audit/Bias_scan/blob/master/Technical_documentation_bias_scan.pdf). The documentation relating to all case studies carried out by Algorithm Audit is publicly available in our [case repository](https://www.algorithmaudit.eu/cases/). In this way, we enable policy makers, journalists, data subjects and other stakeholders to review the normative judgements issued by the audit commissions of Algorithm Audit. 

In sum, our two-pronged approach combines the power of rigorous, machine learning-informed quantitative testing with the balanced judgement of human experts, in order to determine fair AI on a case-by-case basis.

<sup>1</sup><sub>The bias scan tool is based on the k-means Hierarchical Bias-Aware Clustering (HBAC) method as described in Misztal-Radecka, Indurkya, Information Processing and Management. Bias-Aware Hierarchical Clustering for detecting the discriminated groups of users in recommendation systems (2021). Additional research indicates that k-means HBAC, in comparison to other clustering algorithms, works best to detect bias in real-world datasets.</sub>

## Input data bias scan tool
☁️ The tool is available as a web application on the [website](https://www.algorithmaudit.eu/bias_scan/) of Algorithm Audit.

A .csv file of max. 1GB, with columns: features, predicted labels (named: 'pred_label'), truth labels (named: 'truth_label'). Note: Only the naming, not the order of the columns is of importance. The following dataframe is digestible by the web application:

| feat_1 | feat_2 | ... | feat_n | pred_label | truth_label |
|--------|--------|-----|--------|------------|-------------|
| 10     | 1      | ... | 0.1    | 1          | 1           |
| 20     | 2      | ... | 0.2    | 1          | 0           |
| 30     | 3      | ... | 0.3    | 0          | 0           |
<sub>*Table 1 – Structure of input data in the bias scan web application*</sub>

Note that the features values are unscaled numeric values, and 0 or 1 for the predicted and ground truth labels.

## Case study – BERT disinformation classifier (Twitter1516 data set)
We use the unsupervised bias scan tool to assess fair treatment of a self-trained disinformation detection algorithm on the Twitter1516 dataset. Below, statistically significant disparities found by the tool are presented. Based on these quantitative results, questions are distilled and submitted to an audit commission consiting of AI experts. This audit commission formulates a normative advice if, and how, (higher-dimensional) unfair treatment can be assessed on the basis of the bias scan results.

### Bias scan pipeline
A BERT disinformation classifier is trained on true and false tweets (n=1,057) from the [Twitter1516](https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=0&file_subpath=%2Frumor_detection_acl2017) dataset. For this dataset, user and content features are [collected](https://github.com/NGO-Algorithm-Audit/Bias_scan/blob/master/data/Twitter_dataset/Twitter_API_data_collection.ipynb) from the Twitter API. 

📑 More details on the training process of the BERT disinformation classifier can be found [here](https://github.com/NGO-Algorithm-Audit/Bias_scan/blob/master/classifiers/BERT_disinformation_classifier/BERT_Twitter_classifier.ipynb). 

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

📑 More details on this case study can be found [here](https://github.com/NGO-Algorithm-Audit/Bias_scan/blob/master/HBAC_scan/HBAC_BERT_disinformation_classifier.ipynb). 

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

📑 More details on this case study can be found [here](https://github.com/NGO-Algorithm-Audit/Bias_scan/blob/master/HBAC_scan/HBAC_BERT_disinformation_classifier.ipynb).  

### Audit commission: Qualitative assessment of identified disparities
The above quantitative disparities do not establish prohibited _prima facie_ discrimination. Rather, the identified disparities serve as a starting point to assess potential unfair treatment according to the context-sensitive qualitative doctrine. To assess unfair treatment, we question a commission of experts:
1. Is there an indication that one of the statistically significant features, or a combination of the features, stated in Figure 3-4 are critically linked to one or multiple protected grounds? 
2. In the context of disinformation detection, is it as harmful to classify true content as false (false positive) as false content as true (false negative)?
3. For a specific cluster of people, is it justifiable to have true content classified as false 8 percentage points more often? For a specific cluster of people, is it justifiable to have false content classified as true 13 percentage points more often?
4. Is it justifiable that the disinformation classification algorithm is too harsh towards users with verified profile, more #followers and higher user engagement and too lenient towards users with non-verified profile, less #followers and lower user engagement?

📑 More context on the questions submitted to the audit commission can be found [here](https://github.com/NGO-Algorithm-Audit/Bias_scan/blob/master/audit_commission/Problem%20statement%20disinformation%20detection.pdf). 

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
    └── Technical_documentation_bias_scan.pdf                   # Techical documentation (report)
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
- Selma Muhammad, Trustworthy AI consultant at Deloitte
- Laurens van der Maas, Data Scientist at AWS
- Xiaoming op de Hoek, Data Scientist at Rabobank
- Jan Overgoor, Data Scientist at SPAN

#### Academia
- Anne Meuwese, Professor in Public Law & AI at Leiden University
- Hinda Haned, Professor in Responsible Data Science at University of Amsterdam
- Raphaële Xenidis, Associate Professor in EU law at Sciences Po Paris
- Marlies van Eck, Assistant Professor in Administrative Law & AI at Radboud University
- Emma Beauxis-Ausselet [to be confirmed], Associate Professor Ethical Computing at University of Amsterdam
- Aileen Nielsen, Fellow Law&Tech at ETH Zürich
- Vahid Niamadpour, PhD-candidate in Linguistics at Leiden University
- Floris Holstege, PhD-candidate in Explainable Machine Learning at University of Amsterdam

#### Civil society organisations
- [Progressive Café](https://progressiefcafe.nl), founded by Kiza Magendane
- YY
- Simone Maria Parazzoli, OECD Observatory of Public Sector Innovation (OPSI)