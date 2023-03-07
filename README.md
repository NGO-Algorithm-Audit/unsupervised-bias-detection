![image](./images/Header_Github.png)
## An expert-led, deliberative audit informed by a quantitative bias scan

📊 Main presentation: [slides](https://github.com/NGO-Algorithm-Audit/Bias_scan/blob/master/Main_presentation_joint_fairness_assessment_method.pdf)

📄 Technical documentation: [report](https://github.com/NGO-Algorithm-Audit/Bias_scan/blob/master/Technical_documentation_bias_scan.pdf)

## Key takeaways – Why this approach?
- **Quantitative-qualitative joint method**: Data-driven bias testing combined with the balanced and context-sensitive judgment of human experts;
- **Audit commission**: Expert-led, deliberative assessment to establish unfair treatment;
- **Bias scan tool**: Scalable method based on machine learning to detect algorithmic bias;
- **Unsupervised bias detection**: No user data needed on protected attributes;
- **Detects complex bias**: Identifies unfairly treated groups characterized by mixture of features, detects intersectional bias;
- **Model-agnostic**: Works for all binary AI classifiers;
- **Open-source and not-for-profit**: Easy to use and available for the entire AI auditing community.

## Executive summary
Bias in machine learning models can have far-reaching and detrimental effects. With the increasing use of AI to automate or support decision-making in a wide variety of fields, there is a pressing need for bias assessment methods that take into account both the statistical detection and the social impact of bias in a context-sensitive way. This is why we present a comprehensive, two-pronged approach to addressing algorithmic bias. Our approach comprises two components: (1) a quantitative bias scan tool, and (2) a qualitative, deliberative assessment. In this way, the scalable and data-driven benefits of machine learning work in tandem with the normative and context-sensitive judgment of human experts, in order to determine fair AI in a concrete way.

Our bias scan tool, which forms the quantitative component, aims to discover complex and hidden forms of bias. The tool is specifically geared towards detecting unforeseen forms of bias and higher-dimensional forms of bias. Aside from unfair biases with respect to established protected groups, such as gender, sexual orientation, and race, bias can also occur with respect to non-established and unexpected groups of people. These forms of bias are more difficult to detect for humans, especially when the unfairly treated group is defined by a high-dimensional mixture of features. Our bias scan tool is based on an unsupervised clustering method, which makes it capable of detecting these complex forms of bias. It thereby tackles the difficult problem of detecting proxy-discrimination that stems from unforeseen and higher-dimensional forms of bias, including intersectional forms of discrimination. 

Informed by the quantitative results of our bias scan tool, subject-matter experts carry out an evaluative assessment in a deliberative way. The deliberative assessment provides a deeper understanding and evaluation of the detected bias. Such an assessment is essential for a balanced, normative assessment of the bias and its potential consequences in terms of social impact. This is because factual observation of quantitative discrepancies does not establish discrimination or unfair bias by itself. Instead, quantitative discrepancies can only serve as a starting point for evaluating the possibility of unfair treatment in a qualitative way, where the qualitative assessment takes into account the particular social context and the relevant legal doctrines. Human interpretation is thereby required to establish which statistical disparities do indeed qualify as unfair bias. The diversity of our commission of experts contributes to a context-sensitive and multi-perspectival evaluation of the particular use case under investigation. This expert-led, deliberative approach is commonly used by NGO Algorithm Audit to provide ethical guidance on issues that arise in concrete algorithmic use cases. The results of deliberative assessments are published in a transparent way. We thereby enable policy makers, journalists, data subjects and other stakeholders to review the normative judgements issued by the audit commissions of Algorithm Audit, thereby contributing to public knowledge building on the responsible use of AI.

Regarding the technical properties of our bias scan tool: we developed and implemented our tool as a scalable, user-friendly, free-to-use, and open-source web application. The tool works on all sorts of binary AI classifiers and is therefore model-agnostic. This design choice was intentional, as it enables users to audit the specific classifier applied in their use case, whatever it may be. The tool is based on the k-means Hierarchical Bias-Aware Clustering (HBAC) method as described in recent academic literature<sup>1</sup>. Using this method, the tool is able to identify groups of similar users (called ‘clusters’) that are potentially treated unfairly by a binary AI classifier. Importantly, our bias scan tool does not require _a priori_ information about existing disparities and protected attributes. It thereby allows for detecting possible proxy discrimination, intersectional discrimination and other types of (higher-dimensional) unfair differentiation not easily conceived by humans. The tool identifies clusters of users that face a higher misclassification rate compared to the rest of the data set. The classification metric by which bias is defined can be manually chosen by the user in advance: false negative rate, false positive rate, or accuracy. The tool is available as a [web application](https://www.algorithmaudit.eu/bias_scan/) on the website of Algorithm Audit<sup>2</sup>. It is thereby freely available for use by the AI auditing community.

To evaluate and demonstrate our approach in practice, we applied our joint method on a BERT-based disinformation detection model, which is trained on the widely cited Twitter1516 data set<sup>3</sup>. For the quantitative part, our bias scan identifies a statistically significant bias in misclassification rate against a cluster of similar users (characterized by a verified profile, higher number of followers and higher user engagement score). On average, users belonging to this cluster face more true content being classified as false (false positives). We also find a bias with respect to another cluster of users, for whom false content is more often classified as true (false negatives). Our results are robust against alternative configurations of the classification algorithm. In particular, we re-perform our analysis with 162 different configurations of the hyperparameters of the classification algorithm, yielding over 1,000 different clustering results. Aggregating these results produces the same findings as our main analysis.

Based on our results, we formulated a set of pressing questions about the performance of the disinformation classifier, and presented these questions to an independent audit commission composed of four academic experts on fair AI and two civil society organizations working on disinformation detection. Building on the quantitative results of the bias scan, these experts provided substantiated, normative judgments on whether the classifier is causing unfair treatment or not. The audit commission believes there is a low risk of (higher-dimensional) proxy discrimination by the reviewed BERT-based disinformation classifier and that the particular difference in treatment identified by the quantitative bias scan can be justified, if certain conditions apply. This normative advice is supported by 20+ actors from the AI auditing community, including journalists, civil society organizations, NGOs, corporate data scientists and academics.

The detailed results of our case study and our general approach to bias assessment are visually presented in the attached presentation. In-depth discussion of the technical details can be found in the technical [documentation](https://github.com/NGO-Algorithm-Audit/Bias_scan/blob/master/Technical_documentation_bias_scan.pdf).

In sum, our joint approach combines the power of rigorous, machine learning-informed quantitative testing with the balanced judgment of human experts, in order to determine fair AI on a case-by-case basis.

<sub><sup>1</sup>The bias scan tool is based on the k-means Hierarchical Bias-Aware Clustering method as described in Bias-Aware Hierarchical Clustering for detecting the discriminated groups of users in recommendation systems, Misztal-Radecka, Indurkya, _Information Processing and Management_ (2021). [[link]](https://www.sciencedirect.com/science/article/abs/pii/S0306457321000285) Additional research indicates that k-means HBAC, in comparison to other clustering algorithms, works best to detect bias in real-world datasets.</sub>

<sub><sup>2</sup>The uploaded data is instantly deleted from the server after being processed.</sub>

<sub><sup>3</sup>Real-time Rumor Debunking on Twitter, Liu et al., _Proceedings of the 24th ACM International on Conference on Information and Knowledge Management_ (2015).</sub>

## Bias scan tool manual
☁️ The tool is available as a web application on the [website](https://www.algorithmaudit.eu/bias_scan/) of Algorithm Audit.

A .csv file of max. 1GB, with columns: features, predicted labels (named: 'pred_label'), ground truth labels (named: 'truth_label'). Note: Only the naming, not the order of the columns is of importance. The dataframe displayed in Table 1 is digestible by the web application.

| feat_1 | feat_2 | ... | feat_n | pred_label | truth_label |
|--------|--------|-----|--------|------------|-------------|
| 10     | 1      | ... | 0.1    | 1          | 1           |
| 20     | 2      | ... | 0.2    | 1          | 0           |
| 30     | 3      | ... | 0.3    | 0          | 0           |


<sub>*Table 1 – Structure of input data in the bias scan web application*</sub>

Note that the features values are unscaled numeric values, and 0 or 1 for the predicted and ground truth labels.

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
    ├── Main_presentation_joint_fairness_assessment_method.pdf  # Main presentation (slides)
    ├── README.md                                               # Readme file 
    └── Technical_documentation_bias_scan.pdf                   # Techical documentation (report)
```
## Contributors and endorsements
### Algorithm Audit's bias scan tool team:
- Jurriaan Parie, Trustworthy AI consultant at Deloitte
- Ariën Voogt, PhD-candidate in Philosophy at Protestant Theological University of Amsterdam
- Joel Persson, PhD-candidate in Applied Data Science at ETH Zürich

### 20+ endorsements from various parts of the AI auditing community 
#### Journalism
- Gabriel Geiger, Investigative Reporter Algorithms and Automated Decision-Making at Lighthouse Reports

#### Civil society organisations
- [Maldita](https://maldita.es/maldita-es-journalism-to-not-be-fooled/), an independent journalistic platform focused on the control of disinformation and public discourse through fact-checking and data journalism techniques
- [Demos](https://demos.co.uk/), Britain's leading cross-party think-tank
- [AI Forensics](https://www.aiforensics.org), a European non-profit that investigates influential and opaque algorithms
- [NLAIC](https://nlaic.com), The Netherlands AI Coalition
- [Progressive Café](https://progressiefcafe.nl), public platform of young Dutch intellectuals, represented by Kiza Magendane
- [Dutch AI Ethics Community](https://www.linkedin.com/company/daiec/), represented by Samaa Mohammad
- Simone Maria Parazzoli, OECD Observatory of Public Sector Innovation (OPSI)

#### Industry
- Selma Muhammad, Trustworthy AI consultant at Deloitte
- Laurens van der Maas, Data Scientist at AWS
- Xiaoming op de Hoek, Data Scientist at Rabobank
- Jan Overgoor, Data Scientist at SPAN
- Dasha Simons, Trustworthy AI consultant at IBM

#### Academia
- Anne Meuwese, Professor in Public Law & AI at Leiden University
- Hinda Haned, Professor in Responsible Data Science at University of Amsterdam
- Raphaële Xenidis, Associate Professor in EU law at Sciences Po Paris
- Marlies van Eck, Assistant Professor in Administrative Law & AI at Radboud University
- Aileen Nielsen, Fellow Law&Tech at ETH Zürich
- Vahid Niamadpour, PhD-candidate in Linguistics at Leiden University
- Ola Al Khatib, PhD-candidate in the legal regulation of algorithmic decision-making at Utrecht University
- Floris Holstege, PhD-candidate in Explainable Machine Learning at University of Amsterdam

