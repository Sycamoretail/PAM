# Rebuttal of Online Item Cold-Start Recommendation with Popularity-Aware Meta-Learning, KDD 2025, Submission 195

Thank all the reviewers for their constructive and valuable feedback! We have found all the suggestions very relevant and will definitely incorporate them into the paper. 

Due to character limitations, we have included all of the experimental sections as well as some of the more detailed rebuttal explanations in the attached PDF.

We uploaded the PAM code synchronously to ensure reproducibility of the paper. 

To run PAM, you need to install packages ``tensorflow==1.15.0``, ``pandas`` and ``numpy``, unzip the zipped dataset file ``datasets.zip``, cd into PAM-F folder and run:

```python
python train_ml.py
```

The model will read the pre-stored parameters at the end of period 28, train on period 29 data and evaluate on period 30 data, the results of which will be output and stored.

We also revised the article in response to the reviewer's comments and uploaded the revised version of the image along with the article.

We hope we adequately addressed all the reviewers' concerns and that you would consider reflecting that in your score. If you have any additional questions, we would be happy to answer them during the discussion period =)
