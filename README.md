# GeneExpressionPSAvsPSO
**Differential gene Expression Analysis to distinguish Psoriatic Arthritis in patients with Cutaneous Psoriasis: A Machine Learning Approach**

**Background:**
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

