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
