According to my thesis advancement, I'm emailing to you to say that: 

I ended up planning what to do to build an intelligent infrared video camera.
I understood recent deep learning methods and their optimized versions,
(A review of supervised and self supervised Transformers ), the tools to write the codes by pytorch and the complete method to train, evaluate a system. I realize to find the built in functions or modules to be explored to get rid of writing a code from scratch.

Currently, I have all the focus on how to prepare the source code for our work.

Generally, the initial phase is the most difficult of the work:
- Understanding the dataset, extracting the elements we need from the multiple annotation filles, knowing the paths where dataset is and indexing Labels with integers.
- Data collating, preprocessing, data sampling, Building model code  for training and evaluation and an optimizer for backpropagation, data loading for training the model, training and checkpoint logging.

What i did:

I finished putting LSOTB ready for the training from scratch to test the detection  task of infrared objects in  images and videos by the Detr framework.



Now, I'm exploring :


- Grounded situation recognition with Transformers for images GSRTr and Video object grounding VOGNet for videos, action recognition tasks and identiying entities engaged in the action and their roles based on ActivityNet entities dataset.

-  AttentionGAN to transform ActivityNet entities dataset to infrared mode.


- retrain  it to obtain features directly coming out from infrared mode.

So,  what i aim to do for these two months is to finish my code source for the 3 aforementionned tasks.


References:
-LSOTB-TIR: A large scale high diversity thermal Infrared object tracking Benchmark 2020.
-End to End object detection with Transformers2021.
- Grounded situation recognition 2020.
-Vide object Groundong using semantic Roles on Language Description 2020.
AttentionGAN: converting optical videos to infrared Videos using Attention GAN and its impact on target Detection and classification performance 2021.


Then, soon:

- We have  to scale up the work  for long term  dependent video.

- We will see how a good feature representation  can be obtained from the data  without annotation with self supervised Transformers.

-Then,  we will search for other  optimization tricks and theories from neuroscience so that the software can be  mounted on camera with limited computational resources and is endowed with very high intelligence.

-The result of predicting whether an intrusion has taken place has an impact on human life, so we will seek how to make the system interpretable and explainable by, for example, reformulating it with knowledge graphs based on transformer modules to have the prediction compréhensive by system user.

-We will see how to enter other modality such as skeleton data  from Posetics datset  and depth information  from NTu egb+d ans PKU MMD to improve result accuracy.
Best regards.
