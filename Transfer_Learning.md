# Awesome Transfer Learning

In this markdown, there is a list of Transfer Learning material mostly for Natural Language Processing, divided into categories by the type of resource.



## Theory Papers:
[How Transferable are Neural Networks in NLP Applications?](https://arxiv.org/pdf/1411.1792.pdf) (2016)
They tried to follow a similar setting with CNN for images while transfer to encoder-decoder frameworks. In the paper, they focus on two issues: (1)the specialization of higher layer neurons to their original task at the expense of performance on the
target task, which was expected, and (2) optimization difficulties related to splitting networks between co-adapted neurons, which was not expected.

[A Pilot Study of Domain Adaptation Effect for Neural Abstractive Summarization](https://arxiv.org/abs/1707.07062) (2017, working on summarization)
This is a pilot work for domain adaptation+abstractive summarization. The model is simple but the analysis experiments are solid. 


## Recent Papers
[DARLA: Improving Zero-Shot Transfer in Reinforcement Learning](https://arxiv.org/abs/1707.08475) (ICML 17’)
This paper proposed a new multi-stage RL agent to zero-shot transfer learning tasks. The model significantly outperforms conventional baselines in computer vision.


[Supervised and Unsupervised Transfer Learning for Question Answering](https://arxiv.org/abs/1711.05345) (Naacl 18’) [code](https://github.com/chun5212021202/QACNN)
They proposed supervised and unsupervised methods for question answering based on three different datasets: TOEFL, MCTest and MovieQA. The model improves the performance on TOEFL dataset by 7%. They show that transfer learning is helpfull in an unsupervised learning setting. 


[NLP Using millions of emoji occurrences to learn any-domain representations for detecting sentiment, emotion and sarcasm](http://www.aclweb.org/anthology/D17-1169) (EMNLP 17’)



## EMNLP 17
[Learning to select data for transfer learning with Bayesian Optimization](https://www.aclweb.org/anthology/D17-1038)

[Cross-Lingual Induction and Transfer of Verb Classes Based on Word Vector Space Specialisation](http://aclweb.org/anthology/D17-1270)

[Cross-Lingual Transfer Learning for POS Tagging without Cross-Lingual Resources](https://aclanthology.coli.uni-saarland.de/papers/D17-1302/d17-1302)

[Two-Stage Synthesis Networks for Transfer Learning in Machine Comprehension](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/07/emnlp17_SynNet.pdf)

## ACL 2018
[Recursive Neural Structural Correspondence Network for Cross-domain Aspect and Opinion Co-Extraction](http://aclweb.org/anthology/P18-1202)

[Strong Baselines for Neural Semi-supervised Learning under Domain Shift](https://aclanthology.info/papers/P18-1096/p18-1096)

[Domain Adaptation with Adversarial Training and Graph Embeddings](https://arxiv.org/pdf/1805.05151.pdf)

[Two Methods for Domain Adaptation of Bilingual Tasks: Delightfully Simple and Broadly Applicable](https://acl2018.org/paper/593/)

[Domain Adapted Word Embeddings for Improved Sentiment Classification](https://arxiv.org/pdf/1805.04576.pdf)

[Zero-Shot Transfer Learning for Event Extraction](https://drive.google.com/file/d/1jfRQEo3RvubwmmnNLa5vX7_DbtwwX7L_/view)

[Identifying Transferable Information Across Domains for Cross-domain Sentiment Classification](http://aclweb.org/anthology/P18-1089)

[A Helping Hand: Transfer Learning for Deep Sentiment Analysis](http://aclweb.org/anthology/P18-1235)

[Transfer Learning for Context-Aware Question Matching in Information-seeking Conversation Systems in E-commerce](http://aclweb.org/anthology/P18-2034)

[Asymmetric Tri-training for Unsupervised Domain Adaptation](https://arxiv.org/abs/1702.08400)


## Codes & Papers:
[Transfer Learning for Speech and Language Processing](https://arxiv.org/pdf/1511.06066.pdf) (2015)

[Google's Multilingual Neural Machine Translation System: Enabling Zero-Shot Translation](https://arxiv.org/pdf/1611.04558.pdf) (2016)

[Transfer Learning for Low-Resource Neural Machine Translation](https://aclweb.org/anthology/D16-1163)

[TransNets: Learning to Transform for Recommendation](https://arxiv.org/pdf/1704.02298.pdf) (2017)

[A Practitioners’ Guide to Transfer Learning for Text Classification using Convolutional Neural Networks](https://scirate.com/arxiv/1801.06480) (2018)

## Presentation Slides and Talks
http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2017/Lecture/transfer.pdf

https://epat2014.sciencesconf.org/conference/epat2014/pages/slides_DA_epat_17.pdf


https://simons.berkeley.edu/talks/trevor-darrell-2017-3-29 (Video)



## Thesis:
[Deep Learning Models for Unsupervised and Transfer Learning PhD Thesis](http://www.cs.toronto.edu/~nitish/nitish_thesis.pdf), University of Toronto, May 2017

[Transfer Learning Techniques for Deep Neural Nets](http://www.cs.utep.edu/ofuentes/theses/Gutstein_Dissertation.pdf) , Steven Michael Gutstein, 2010

[Feature-based Transfer Learning and Real-world applications](https://pdfs.semanticscholar.org/171c/0aa92b49e27f661a9cb1dd990d2f529d21da.pdf), Sinno Jialin Pan

## Survey:
http://www.jmlr.org/papers/volume10/taylor09a/taylor09a.pdf (with Reinforcement Learning 2009)

https://www.cse.ust.hk/~qyang/Docs/2009/tkde_transfer_learning.pdf (2009)

https://arxiv.org/pdf/1705.04396.pdf (2017)

https://arxiv.org/pdf/1707.08114.pdf (2017) (Multi-task Learning)

https://journalofbigdata.springeropen.com/articles/10.1186/s40537-016-0043-6 (2015-2016)

http://www.umiacs.umd.edu/~pvishalm/Journal_pub/SPM_DA_v9.pdf (2015)

## Resources, materials and interesting ideas:
http://tommasit.wixsite.com/datl14tutorial/bibliography (2014, list of papers, tutorials)

[Fighting Offensive Language on Social Media with Unsupervised Text Style Transfer](https://arxiv.org/pdf/1805.07685.pdf)

[Deep Text Style Transfer](http://www.cs.tau.ac.il/~joberant/teaching/advanced_nlp_spring_2018/past_projects/style_transfer.pdf)

[Transfer Learning in NLP](https://blog.feedly.com/transfer-learning-in-nlp/) (2018)
