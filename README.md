![image](./images/Header_Github.png)
## Detecting higher-dimensional form of proxy bias

📄 Applied in real-world audit: [audit report](https://algorithmaudit.eu/algoprudence/cases/aa202402_preventing-prejudice_addendum/)

📊 Main presentation: [slides](https://github.com/NGO-Algorithm-Audit/Bias_scan/blob/master/Main_presentation_joint_fairness_assessment_method.pdf)

📄 Technical documentation: [report](https://github.com/NGO-Algorithm-Audit/Bias_scan/blob/master/Technical_documentation_bias_scan.pdf)

## Key takeaways – Why this approach?
- **Quantitative-qualitative joint method**: Data-driven bias testing combined with the balanced and context-sensitive judgment of human experts;
- **Normative advice commission**: Expert-led, deliberative assessment to establish unfair treatment;
- **Bias scan tool**: Scalable method based on machine learning to detect algorithmic bias;
- **Unsupervised bias detection**: No user data needed on protected attributes;
- **Detects complex bias**: Identifies unfairly treated groups characterized by mixture of features, detects intersectional bias;
- **Model-agnostic**: Works for all binary AI classifiers;
- **Open-source and not-for-profit**: Easy to use and available for the entire AI auditing community.

## New
...

<sub><sup>1</sup>The bias scan tool is based on the k-means Hierarchical Bias-Aware Clustering method as described in Bias-Aware Hierarchical Clustering for detecting the discriminated groups of users in recommendation systems, Misztal-Radecka, Indurkya, _Information Processing and Management_ (2021). [[link]](https://www.sciencedirect.com/science/article/abs/pii/S0306457321000285) Additional research indicates that k-means HBAC, in comparison to other clustering algorithms, works best to detect bias in real-world datasets.</sub>

<sub><sup>2</sup>The uploaded data is instantly deleted from the server after being processed.</sub>

<sub><sup>3</sup>Real-time Rumor Debunking on Twitter, Liu et al., _Proceedings of the 24th ACM International on Conference on Information and Knowledge Management_ (2015).</sub>

## Bias scan tool manual
☁️ The tool is available as a web application on the [website](https://algorithmaudit.eu/technical-tools/bdt/) of Algorithm Audit.

A .csv file of max. 1GB, with columns: features, predicted labels (named: 'pred_label'), ground truth labels (named: 'truth_label'). Note: Only the naming, not the order of the columns is of importance. The dataframe displayed in Table 1 is digestible by the web application.

| feat_1 | feat_2 | ... | feat_n | pred_label | truth_label |
|--------|--------|-----|--------|------------|-------------|
| 10     | 1      | ... | 0.1    | 1          | 1           |
| 20     | 2      | ... | 0.2    | 1          | 0           |
| 30     | 3      | ... | 0.3    | 0          | 0           |


<sub>*Table 1 – Structure of input data in the bias scan web application*</sub>

Note that the features values are unscaled numeric values, and 0 or 1 for the predicted and ground truth labels.

## Contributing Members
- Floris  | https://github.com/fholstege
- Joel Persson | https://github.com/jopersson
- Jurriaan | https://github.com/jfparie
- Kirtan | https://github.com/kirtanp
- Krsto | https://github.com/krstopro
- Mackenzie Jorgensen | https://github.com/mjorgen1

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

