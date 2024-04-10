# Rebuttal of Online Item Cold-Start Recommendation with Popularity-Aware Meta-Learning, KDD 2024, Submission 123

## Reviewer XxHZ:

**Q4: Lack the comparision of latest works.**

Thanks for pointing that out. The field of online recommendations and incremental updates to recommendation models is a small research field with less work. We have selected a latest work in 2023 IMSR[2] as a baseline for comparison. The results of the experiment are as follows:

|Methods|Recall@20|NDCG@20|Impr.%|
|:--:|:--:|:--:|:--:|
|IMSR|XX|XX|-|
|PAM-F|XX|XX|+XX%|

[1] Ma et al. Cross-Modal Content Inference and Feature Enrichment for Cold-Start Recommendation. IJCNN '23
[2] Wang et al. Incremental Learning for Multi-Interest Sequential Recommendation. ICDE 2023





## Reviewer myQM:

**Q1: Categotization of cold-start problems.**

As a few-shot learning method, meta-learning aim to learn the ability to fine-tune a personalized parameters from a small amount of samples. Therefore, samples is necessary for a meta-learning method, without which there is no fine-tuning to generate personalized parameters, and thus the meta-learning method is not able to handle strict cold-start scenarios.

In the start of the system, all items are cold like you said. However, item's popularity gradually increased over time, and after entering the testing phase, we calculated metrics for all cold-start items, not all strictly cold-start items with 0 popularity. Besides, our definition for cold-start items is based on the popularity threshold, which is a common way of defining cold-start items (*e.g.* "divide the cold-start set with few interactions" in [1] and "As for the items, we regard items with less than 10 ratings as cold items" in [2]). Our evaluation metrics are also suitable and wide applied to cold-start items, as it evaluate the rank of candidate users to an item.

[1] Ma et al. Cross-Modal Content Inference and Feature Enrichment for Cold-Start Recommendation. IJCNN '23
[2] Dong et al. MAMO: Memory-Augmented Meta-Optimization for Cold-start Recommendation. KDD '20
