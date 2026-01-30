# zenthos_garvit
The tfrecord file of all the videos is at 
**s3://tensorflow-file-of-charades-dataset/charades_frames_fast_all_videos.tfrecord**


The frame wise embeddings of all the video created using image encoder of clip-vit-L14  are at 
s3://tensorflow-file-of-charades-dataset/charades_clip_embeddings_final/


**mamba-charades-training-code.py**_ was used to train the model on multiple descriptive labels present for a video 40 epochs just by varying the number of epochs 
then later it was finetuned on descriptive labels using the code **description-based-training.py** for 10 epoch 
the checkpoints of this model are saved in s3 at:-
**s3://tensorflow-file-of-charades-dataset/mamba_charades_retraining/**

The model trained again due to low similarity scores of above was trained initally for 10 epochs using 
description-scratch-training-10epoch.py
then for 10 more epochs using 
description-training-10-20-epochs.py
then for 20 more epoch using 
description-training-20-40epoch.py
all the checkpoints are saved in s3 at** s3://tensorflow-file-of-charades-dataset/mamba_charades_10epochs/**
The testing of the model was done in notebook using just by loading differnet checkpoints **10_epoch_fresh_testing_code.py**
In the testing code above i have mentioned all the commands i used to install libraries


All daily updates were added in the googledocs along with related codes , weights and hugging face links 
https://docs.google.com/document/d/1EmNNoWiVy2PoXx5M1QeT9jKGL1eKU17Fbk3Nb2W90Ks/edit?usp=sharing







