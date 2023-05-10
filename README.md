# GeneExpressionPSAvsPSO
Differential gene Expression Analysis to distinguish Psoriatic Arthritis in patients with Cutaneous Psoriasis: A Machine Learning Approach
Background:
Psoriatic arthritis (PSA) is a chronic condition that affects around 30% of individuals with
psoriasis, an inflammatory skin disease[1]. It's important to note that 90% of patients with
psoriatic arthritis have psoriasis and if they don’t have psoriasis there is usually a family member
who does, which highlights the genetic nature of psoriasis and psoriatic arthritis. While both
psoriasis and PSA worsen over time, diagnosing PSA when psoriasis is present has been a
challenge. A study showed that only 23% of patients were diagnosed with PSA at symptom
onset, with 45% diagnosed after 2 years[2]. Early diagnosis and treatment are crucial in
improving long-term outcomes and preventing joint damage and poor functional outcomes.
Additionally, PSA can present similarly to other autoimmune diseases, and according to
arthritis.org PSA patients have double the risk of developing cardiovascular disease, 43% more
likely to have or develop heart disease and a 23% increased risk of developing conditions that
affect blood flow to the brain[3].Dna expression research has led to effective biologic
medications like taltz(ixekizumab ), which lower the expression of il-17 in patients with
psoriasis, PSA, and other diseases. Further research is needed to better define the differences
between psoriasis, PSA, and other autoimmune diseases, and to isolate biomarkers that have
the potential to revolutionize medical care.
In our analysis, we are exploring the use of gene expression data to develop a machine
learning model that can differentiate between psoriatic arthritis and psoriasis(only). We are
utilizing differential gene expression analysis to improve the accuracy of our model by reducing
the number of dimensions in the data. Developing an accurate diagnostic tool is critical as early
courses of biologics have been shown to greatly reduce patient outcomes (management of
psoriatic arthritis: early diagnosis, monitoring of disease severity and cutting edge therapies). If
a patient can get onto suitable treatment to reduce inflammation early it can help stop
aggressive bone, cartilage and tendon damage that otherwise would be irreversible. Identifying
specific biomarkers associated with PSA can also help us better understand the condition and
develop more targeted treatments. The potential for personalized treatment plans using isolated
biomarkers could also be significant. We hope that this analysis will help show the use of gene
expression on top of rheumatological diagnosis tools like CASPAR[1]. CASPAR alone may not
be ideal as it typically is a lagging indicator as it can only measure inflammatory damage after it
has persisted for a good amount of time. That's why gene expression analysis could aid patients
in getting a more timely diagnosis and reduce their risks of developing other mental disorders
like somatic symptom disorder that commonly goes along with psoriatic patients who take an
average of 2.5 years[1] after symptom development to finally get their diagnosis with PSA.
Although some research has been done using differential gene expression analysis in
order to identify important biological pathways, there has been little to no research done on
using machine learning models on top of differential gene expression analysis in order to detect
psoriatic arthritis in patients with psoriasis. We believe that this kind of analysis may provide a
solution to the problem of late diagnosis of psoriatic arthritis. We suggest that the use of an ‘all
encompassing model’ which includes differential gene expression data of multiple cell types,
patient medical history, CASPAR scores from a rheumatologist, MRI and ultrasound technology
may provide a solution to the late diagnosis problem. Unfortunately, even with the recently
popularized use of machine learning and big data, there has been little to no research done on
the use of machine learning to aid in the diagnostic process of psoriatic arthritis.
This may be because most of the research done is focused almost exclusively on
finding new biological pathways in which biologics can be made to treat and profit immensely
from. Currently the biologics space is a multi-billion dollar industry, with it estimated to reach a
total industry market value of 719.84 billion by 2023[4]. This can be observed as companies like
Abbivie (Stock ticker ABBV) have seen tremendous growth and profits from drugs like humira
which can be solidified by constant stock growth due to increasing profit margins and a ‘wall of
patents’ which allowed ABBV to have 165 patents on a single drug, effectively pushing back the
opportunity for biosimilars in an effort to ‘game’ the biologics industry[5]. This isn’t to say that
biologics aren’t important because they are some of the greatest technology advancements that
the medical field has observed, but it's important to encourage more research around early
diagnosis so patients can start biologics treatments earlier and have improved patient outcomes
and reduced health care costs.
Dataset:
The dataset is public and was posted on September 24, 2014 as a GEO dataset
published on the NIH website under GSE61281[6]. It involves human whole blood samples and
compares transcriptional profiles between psoriatic arthritis (PSA), cutaneous psoriasis without
arthritis (PSO), and unaffected controls. The experiment was conducted using expression
profiling by array. The overall design includes three conditions: PSA, PSO, and unaffected
controls. There were 20 biological replicates each for PSA and PSO and 12 for controls.
Although sampling of inflammatory cells would be more ideal, for example synovial fluid cells as
this is a primary area of disease activity, blood cells could provide more information on the high
comorbidity rate that is apparent in both psoriasis and psoriatic arthritis patients with
cardiovascular diseases and diabetes. On top of an increased rate when compared to controls
of these diseases it has also been shown that smoking can increase the chance of developing
both of these diseases[7]. So, it is definitely of interest for more studies to be down with
cardiovascular specific cells like blood, or even lung cells.
Data exploration:
Initially, we conducted an exploratory analysis to identify any significant observations
using various visualization techniques. A scatter plot was generated to illustrate the distribution
of gene expression values, which indicated a range between -7 and 7. Based on our
observations, it appears that the data had been transformed using a log2(x) method. Although it
is not always advisable to assume data transformation, log2(x) is a common method used in
gene expression data analysis, and the range and distribution of our dataset suggests that it has
been log transformed. This is beneficial since it eliminates the need for data scaling and
normalization. Furthermore, a density plot was constructed to assess the distribution of
individual patient data, which revealed a relatively consistent distribution amongst patients. This
suggests that the data had been previously scaled and normalized before the data was
published.
Finding differentially expressed genes:
Differential gene analysis is a powerful tool for gaining insights into the molecular basis
of various diseases and disorders, and for identifying genes that are dysregulated in affected
individuals. In this study, we utilized the r package limma to isolate genes that were significantly
upregulated or downregulated in psoriasis patients compared to controls. We employed the
log2(fold change) method to identify significant genes, which is a widely-used technique for
detecting differentially expressed genes. We generated a volcano plot to visualize the
distribution of our data and identified the top 30
genes that were dysregulated in psoriasis
patients.
Surprisingly, our analysis did not
identify any genes that were significantly
dysregulated in psoriasis patients compared to
controls. This finding is unexpected given the
increased risk of cardiovascular and metabolic
disorders in psoriasis patients. This observation
is consistent with a previous study on this
dataset that also failed to identify differentially
expressed genes between psoriasis patients and
controls. The authors of that study suggested
that the weak signature of cutaneous disease in
whole blood samples might have been obscured
by the heterogeneity of whole blood. However, they also noted that many of the differentially
expressed genes between psoriatic arthritis and psoriasis without arthritis had smaller changes
in expression in PSO patients compared to controls, suggesting that the more severe phenotype
of PSA is an exacerbation of small gene expression changes that occur in PSO.
This finding raises an interesting question about the relationship between psoriasis and
psoriatic arthritis. It is possible that psoriatic disease represents a spectrum of disease severity
rather than discrete entities, and that the dysregulated genes that we failed to identify in
psoriasis patients are still differentially expressed but to a lesser extent. Indeed, some patients
with psoriatic arthritis do not show signs of psoriasis, and it is possible that many of these cases
go undiagnosed by rheumatologists who do not consider the diagnosis without skin symptoms.
In conclusion, our analysis using limma failed to identify any significantly dysregulated
genes in psoriasis patients compared to controls. This unexpected result raises interesting
questions about the relationship between psoriasis and psoriatic arthritis and highlights the need
for further studies to elucidate the molecular mechanisms underlying these disorders.
Enrichment Analysis:
Following the identification of significantly expressed genes, the next step in gene
expression analysis is to perform enrichment/pathway analysis. Enrichment analysis, also
known as gene list analysis, involves comparing and modeling an entire gene list or just the
significant genes against a database of various biological pathways and important genetic
information. The purpose of this step is to gain a deeper understanding of the biological
processes involved in the significant genes or the ordered list of genes sorted by their p-values.
In this study, we utilized g: profiler to perform enrichment analysis, and identified several
interesting findings related to metabolic disorders that are highly comorbid with psoriasis and
psoriatic arthritis. A bubble chart comparing the different categories was generated and revealed
significant biologic processes and pathways related to protein kinase camp-activated catalytic
subunit alpha, which has been implicated in the development of new target drugs for type 2
diabetes, another common comorbidity associated with psoriatic disease. However, parsing
through this vast and complex information requires significant time and biological expertise.
Therefore, careful examination and interpretation of the data is necessary to fully appreciate the
important insights provided by enrichment analysis.
Modeling phase:
We created a model to differentiate between psoriatic arthritis and psoriasis(only) by
identifying significant genes using gene expression data.Since we weren’t interested in
differentiating between psoriasis and controls we left the controls portion of the data out. In the
modeling phase, we tested various machine learning models such as logistic regression, KNN
classifier, random forest classifier, and MLP neural network. We found that the MLP neural
network had the highest accuracy compared to other models, which was further improved by
isolating only the significantly expressed genes. The accuracy of the logistic regression model
was measured to be an average of 90%. These results are much better than our model where
we did not filter out genes that weren’t significant as the MLP only had an accuracy of 67% in
that case which is 23% less accurate.Therefore, this shows that differential gene expression
analysis is a suitable technique to reduce the dimensionality in our gene expression models
where the data has a high amount of dimensionality and a low amount of data .Our results
suggest that the MLP neural network is a suitable method for modeling and classifying psoriatic
arthritis in patients with psoriasis. However, to further improve on model efficacy it would be
beneficial to combine with information obtained by health care specialists like CASPAR, MRI’S,
ultrasounds of tendons and more medical history in order to find a more complete model that
can better diagnose psoriatic arthritis in patients with psoriasis. This type of ‘all encompassing
model’ has not been suggested as far as we are concerned and it would likely reduce the time it
takes to receive a diagnosis in patients with psoriatic arthritis.
Discussion:
In conclusion, this study aimed to use gene expression data to develop a machine
learning model that can differentiate between patients with only psoriasis and those with
psoriatic arthritis as well. Early diagnosis of PSA in patients with psoriasis is crucial for
improving long-term outcomes and preventing joint damage and poor functional outcomes. The
dataset used in this study involved human whole blood samples and compared transcriptional
profiles between PSA, cutaneous psoriasis without arthritis (PSO), and unaffected controls.
However, the analysis did not identify any genes that were significantly dysregulated in psoriasis
patients compared to controls. This finding is unexpected given the increased risk of
cardiovascular and metabolic disorders in psoriasis patients. The potential use of isolated
biomarkers for personalized treatment plans could be significant, and further research is needed
to better define the differences between psoriasis, PSA, and other autoimmune diseases.
Although sampling of inflammatory cells would be more ideal, blood cells could provide more
information on the high comorbidity rate that is apparent in both psoriasis and PSA patients with
cardiovascular diseases and diabetes. Moreover, using an MLP neural network model after
identifying significantly expressed genes could potentially aid in the diagnosis of PSA in patients
with psoriasis. The model could be used as a tool to predict the likelihood of a patient having
PSA based on their gene expression profile. This could potentially help healthcare professionals
make more accurate and timely diagnosis, leading to better treatment outcomes for patients.
However, further research is needed to validate the use of such models in clinical
settings and likely a larger variety of cell types on top of tests like caspar that a rheumatologist
could perform. Further research is needed in order to provide an earlier diagnosis, one where a
significant amount of joint damage has occurred. Finding a way to get an earlier diagnosis not
only improve PSA patient outcomes but would also lower the burden of a PSA patient on the
healthcare system by reducing the chance of needing surgery and by having less of a chance to
develop comorbidities like metabolic disease, somatic symptom disorder, depression and
cardiovascular disease. We suggest that the use of an ‘all encompassing model’ which includes
differential gene expression data of multiple cell types, patient medical history, CASPAR scores
from a rheumatologist, MRI and ultrasound technology may provide a solution to the late
diagnosis problem. Unfortunately, even with the recently popularized use of machine learning
and big data, there has been little to no research done on the use of machine learning to aid in
the diagnostic process of psoriatic arthritis and we may be the first to suggest such a model to
aid in the diagnosing of psoriatic arthritis.
