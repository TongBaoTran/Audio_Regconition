# Audio_Regconition
![Sound Classification App](https://github.com/TongBaoTran/Audio_Regconition/assets/39885305/54843d79-addf-4859-a107-b465b6a68032)


This project is a part of my Bachelor's thesis in Deep Learning: Transfer Learning and its application in Image and Audio Recognition. 
 You can find the full report of my thesis here: https://drive.google.com/file/d/1lGAB5jtpnha-mk8VDiH_Es_58E-93VZ7/view?usp=drive_link
 The dataset and model trained in the thesis can be found on my Kaggle account: https://www.kaggle.com/trantong/code

 I created a Desktop Application called Sound Detector using Python. The user can upload a short recording or a spectrogram of the audio file. If the input is the audio file, the software will create the spectrogram of the recording. The user can also choose the Color Map for the spectrogram. Another input is the model file, which is the result of the training process. I used Keras file format .h5 file. The label file includes Classes/ Labels used in supervised Learning. This file is usually in CSV format.

 I have included here the sample model file and label file for the dataset ESC-50:ESC-50: Dataset for Environmental Sound Classification. The model types have two types: Single or Multiple. Single model Type will give the one prediction with the highest probability, whereas the "Multiple" will give several possible predictions for the selected sound. 

 # Running the project
 You can download the repository, create a virtual enviroment in which you install necessary libraries such as Numpy, Pandas, PyAudio, Matplotlib, etc, and run the application. 
 I include the Demonstration.mkv file to illustrate how the application looks like and works.
