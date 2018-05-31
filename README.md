# Wikiref
==== WikiRef ====

The Code is written for ComputerScience dataset. For Physics dataset please change the path for dataset in the python files necessary, i.e. set 'dataset_path' variable accordingly.

Initially Execute all the Files in `preprocessing' folder [Note: Execute `preprocessing/build_intralink' before `preprocessing/build_inlink']. Extract `dataset/ComputerScience.tar.gz' and `dataset/Physics.tar.gz' [Note: Extract the ones you need]

== Dataset == 
We have two datsets available, Physics and Computer Science

Please download the dataset, extract it and save it in `dataset/' directory.

Download Computer Science dataset from here : https://drive.google.com/file/d/1jJ7j5c5uqxRrUMHz0-yNHKYvIQHhfTwO/view?usp=sharing
Download Physics dataset from here : https://drive.google.com/file/d/1yu_c3MBQVq6qMlbem1iqmsMZMwZO7Yi9/view?usp=sharing

== Step1 ==
Goto `features/Step1' and execute all the files. You get The features.

Goto `features' execute `create_feature_csv.py'. You get csv file of the features.

Goto `classifier' execute `step1_testing.py'. 

Step 1 Done

== Step2 ==

First you need to install `svm_rank_learn' and `svm_rank_classify'. Download it from here https://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html [Note : Make sure `svm_rank_learn' and `svm_rank_classify' executables are in your search tree]

Goto `features/Step2' and execute `step2_training_features.py'. You got your features.

Goto `classifier' execute `step2_testing.py'.

Step 2 Done

== Python Packages Required ==
scipy
sklearn
gensim
numpy
torch
jellyfish
