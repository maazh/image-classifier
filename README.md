# Image Classifier application

### Deep Learning, Transfer Learning

The following project consists of an Image Classifier Application which takes in images of flower and predicts its name along with giving the top 5 predicitions and their probability. 

The application consists of the following files:

(1) train.py - will train a new network on a dataset and save the model as a checkpoint

(2) test.py -  uses a trained network to predict the class for an input image. 

(3) Image-classifier.ipynb / Image-classifier.html - documents a step by step approch of the entire data cleaning, processing, training, tuning and testing steps. 

The application allows the user to set:

For training:
- Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
- Choose architecture: python train.py data_dir --arch "vgg13"
- Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
- Use GPU for training: python train.py data_dir --gpu

For testing:
- Turn top KK most likely classes: python test.py input checkpoint --top_k 3
- Use a mapping of categories to real names: python test.py input checkpoint --category_names cat_to_name.json
- Use GPU for inference: python test.py input checkpoint --gpu

