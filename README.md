
# What is this project?

This project is an educational resource that wishes to make learning about Artificial Intelligence more easy and provide a platform for anyone to create deep convolutional networks in seconds without having to know any programming syntax! It is an effective tool for testing and prototyping real world projects based on deep CNNs as well!  In the project's current state, one can only make networks to classify music according to genres of music but I will expand this project in the future to encompass more Artificial Intelligence algorithms.

## How does it achieve the goal?
The program is a tool for learning about deep convolutional neural networks by simplifying the process making and training models significantly. Many people wish to learn about neural networks but they do not have any prior experience in programming which makes neural networks and other artificial intelligence algorithms all the more daunting. The goal of this project is to increase the accessibility of the knowledge of Artificial Intelligence algorithms by removing the prerequisite of knowing programming syntax completely. As the prerequisite of programming is removed, more people would be interested and  therefore more people can learn about fascinating AI algorithms.

How is this done? Well it is done by making the entire process of crafting and training neural networks entirely graphical. The user can add layers, remove them, change parameters, all by the clicks of a few buttons. Once trained all the stats of the custom model that the user has made are shown. The user can then play around with his custom model and predict new data or play a game with their model by trying to see who can get the most guesses right while predicting the genre of music clips.

## Inspiration:
I have always wanted to teach people about computers. Artificial intelligence is a field that fascinates me and I wish to make it just as fascinating for other people by making it easier for them to understand it through simple interactive educational resources such as this project.

The inspiration behind this project heavily lies on:
1. tensorflow's playground `playground.tensorflow.org`
2. Harvard CS50's Introduction to Artificial Intelligence with python `https://cs50.harvard.edu/summer/ai/2020/`
3.  The YouTube channel 3Blue1Brown
 `https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw`
4. Me and my best friend, Judica's, love for music and for teaching people!


# How to get started?
First, make sure you have a dataset of music and their labels; the more data, the better! If you do not have a dataset then visit :
>http://marsyas.info/downloads/datasets.html

and download a dataset from there. Note: the dataset should be structured in a way that there is an outer directory encompassing other directories which are named after the genre of the files in those respective directories. The algorithm preprocessing the data will label the music files after the directory the music file is in so make sure to get that right!

Next, download all the files from this page. Make sure that you at least download `prepareData.py`, `genre.py` , `main.py` and  `requirements.txt` as these are the most important files. 

Once all of these files are downloaded, it is preferable to move all of them into one directory with the dataset.

Next, open up Terminal and go to the directory where all these files are saved by using `cd absolute/directory/of/the/files`

After that run `pip3 install -r requirements.txt` in the terminal

All the installation part is now done :)

## Running the program
Once again go to the directory of the files in your terminal. Now type in the following command: `python3 main.py` or `python main.py`. 
>Warning: Make sure that the files `prepareData.py`, `genre.py` and `main.py` are in the same directory or else this won't work. If any error pops up stating that a particular library/package was not installed then please install it using `pip3 install packageName` or  `pip install packageName` as I might have forgotten something in the requirements.


Typing in the above mentioned command should open up a simple and interactive GUI. 



### Data Pre-Processing

This is the first screen of the program. Here you are greeted with a couple of frames.

1. The  `Import Raw Data` frame: 
`Import Dataset` Import raw music clips and process them to make them suitable to be fed into a CNN.
	`Save Data After Processing` Save the data into a json file after it is processed so that you don't need to worry about processing it again the next time you with to use this program again.

2. The  `Import Processed Data` frame:
`Import Processed Data` Import a json file consisting of all the processed data.

3. The  `Deep Convolutional Neural Network` frame:
`Tune Data Processing` Change the parameters of processing your data.
`Go to Convolutional Neural Network` Takes you to the next page where you build your own CNN.


### Building Deep Convolutional Networks 
This is the second screen of the program on the same root screen. Here you are greeted with a couple of frames .

1. The  `Build your own Deep Donvolutional Neural Network` frame:
`Add a Convolution and Pooling Layer` Add a convolution and pooling layer to your custom neural network after customizing the parameters of the layer.
`Add a Dense Hidden Layer` Add a dense hidden to your custom neural network after customizing the parameters of the layer.
`Remove the Top Layer` Remove the top layer regardless of it being convolution or dense hidden layer.
	>Note: All the convolution layers are put first in the layer followed by the hidden layers therefore the intertwining of both the layers in the GUI will not cause any problems :) 
	
2.  The  `Adjust Training Parameters` frame:
`Adjust Training Hyper Parameters` Customize the parameters that are to be used to train your custom neural network.
`Ajust Early Stopping` Customize the parameters that are used to call the early stopping callback.
	>Note: 
		1. If you wish to remove the early stopping entirely then you can do so in `Adjust Training Hyper Parameters` . 
			2. Adding any parameters will show up in the GUI but will not be removable. This is because these parameters can be overwritten and if you made a mistake then just add it again and the program will consider the newest parameters.
			
3. The `choose model` frame:
	 `Train the Custom Model` Build and train the custom model the user has provided. 
	  `Train the Default Model` Build and train the vanilla model that has been provided by me.
	   `Use Saved Model` Import a previously saved model.

	
Keep in mind that every time you add a layer or add some parameters, it shows up at the bottom in the  `Your custom Deep Convolutional Neural Network`  frame along with the previous additions so that you can keep a track of the structure of your custom deep convolutional neural network.

Once you click on one of the buttons in the choose model frame, you need to wait for a while till your model is built and trained or imported. You can monitor the progress of your model by visiting the Terminal and watch it being trained, you will also see the structure of this model once it is trained.

### Testing 
Once your model has been built and trained/imported, you will be taken to this screen.

Above in the `Your custom Deep Convolutional Neural Network` frame, your model is shown.

Next is the `Test the Model` frame:
In this frame there are three more Frames:
1. The `Model Statistics` frame:
	The frame shows the accuracy and the loss of the model.
2. The `Model Charts` frame:
	`View Charts of the Model` View the accuracy and loss charts of the model plotted with respect to epochs.
	>Note: If the user has imported a previously saved model then the charts won't be shown
	
3. The `Play around with the Model` frame: 
	`Predict` Asks for a file location and name. Predicts the genre of the clip specified.
	`Play a guessing game against the AI` Asks for a directory of music. Picks 5 random songs from the directory. The model predicts on these songs. The user is given an opportunity to listen to the clips and type in what genre they think it is in the input fields. Once the user clicks `Get Results`, the results are shown to the user, comparing their prediction to their AI's prediction. `Exit` Exits the screen.
4. The `Save the Model` frame:
	`Save Model` Save the model for future.
	`Exit` Exits the program.

## The brains behind this
The brains behind all this are the three files: `prepareData.py`, `genre.py` , and `main.py` . `prepareData.py` and `genre.py` are throughly commented to make it easy for you to understand if you wish to do so. `main.py` has less comments which I apologize for as I did not have enough time to comment it properly. It is still very understandable though.  

Anyone is welcome to modify these files to suit it for their purpose or make it better.

# Conclusion
Unfortunately this is the end of the program. I wish to further expand it if it get the time and resources. My goal with this project was to make learning artificial intelligence more accessible for everyone. I believe that with this interactive way of learning about deep convolutional neural network through this project has allowed me to move towards the goal. 
This is just a small step in the direction of my goal, in the future I will be putting out many more interactive educational resources regarding Artificial Intelligence to teach people about AI and how amazingly fascinating it is. My life goal is to become a professor at Harvard University and teach people about Artificial Intelligence with my lectures and interactive educational resources. I believe that I will be able to achieve it in the end no matter what.

I would like to give huge thanks to Brian Yu, The professor teaching CS50's Introduction to Artificial Intelligence with python. Without him and his amazing lectures this project or my goals would not have been possible. Moreover I would like to thank my TF, Connor Legget, as well. He guided me through this project and was very supportive. I could not have asked for a better person to support this project. In the end, I am thankful to Brian and Connor for giving me this opportunity to do this project. I had a lot of fun doing this and in the process discovered myself and my goals which is more than what anyone could ask for.

I hope that this project is able to help people understand better and more easily about fascinating Artificial Intelligence algorithms.

Cheers!












