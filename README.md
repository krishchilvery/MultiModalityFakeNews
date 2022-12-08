# MultiModalityFakeNews
Comparison of Algorithms, Augmentation and Multimodality

## Setting up the code
1. Increase python version 3.9 in the system
2. Navigate to project directory and run "pip install -r requirements.txt"

### Running Multi Modality Fake News Code
1. Download "multimodal_only_samples.zip" from https://drive.google.com/drive/folders/1jU7qgDqU1je9Y0PMKJ_f31yXRo5uWGFm and extract the .tsv files to a new folder named "multimodal_data" folder in your project directory.
2. Initialize the jupyter notebook using "python -m notebook"
3. Find multimodalityfakenews.ipynb in the resulting browser window and open it in the notebook window.
4. To just test the working of the code use the training and val sizes as 500 and 50 respectively for faster computation. If accuracy is the goal increase the train size to 50000 and validation size to 5000. This can be done by tuning the variables TRAIN_SIZE and VAL_SIZE
5. Run cell by cell to train the model and test the accuracy.


### Running the code for unimodal models
1. Download "all_samples.zip" from https://drive.google.com/drive/folders/1jU7qgDqU1je9Y0PMKJ_f31yXRo5uWGFm and extract the .tsv files to a new folder
2. For the basic_models.py file, open it in an editor and modify the path to the .tsv files on lines 31, 33, and 35 and then run the code
3. For the cnn.py file, open it in an editor and modify the path to the .tsv files on lines 39, 41, and 43 and the path to the GloVe embeddings on line 216 and then run the code.
4. For the aug.py file, open it in an editor and modify the path to the .tsv files on lines 25, 26, and 27 and the openai api key on line 145. This can be obtained by creating an account on openai and copying the user's api key. Running this code will give 3 files as output, 'bt_augmentation.csv' containing the augmented examples using back translation along with the original data, 'gpt_augmentation.csv' containing the augmented examples using gpt3 along with the original data, and 'bt_gpt_augmentation.csv' containing the both the back translation and gpt3 augmented examples along with the original data. To test the above unimodal models with the augmented data, just change the path of the training data file on line 31 of basic_models.py and line 39 of cnn.py
5. Note: 20,000 samples each for back translation and gpt3 might take a lot of time, so the user can change the number of samples to be augmented on line 72 
