from baseObjects import *
import pathlib, sys, time, pygame
import matplotlib.pyplot as plt
import genre, prepareData

activationFuncs = ["deserialize", "elu", "exponential", "get", "hard_sigmoid", "linear", 
"relu", "selu", "serialize", "sigmoid", "softmax", "softplus", "softsign", "swish", "tanh"
]
padding = ["valid", "same"]

convHP = []
hiddenLayerHP = []

"""
////////////////
Main Root Screen
////////////////
"""
class rootScreen():
    """
    This is the Root screen class
    It is the starting and main screen
    """
    
    #intializing root screen 
    root = Tk()
    root.title("Maple Music Genre Classifier")
    root.geometry("1000x675")
    objects = genericViewObjects(root)
    rowCounter = 0
    
    def __init__(self):
        """
        Initialize
        """

        self.show()

    def show(self):
        """
        Display all the immediate objects to be displayed to the user on initiation
        """

        #defining the frames
        rootFrame = self.objects.frame("Welcome to Maple's free Deep Convolutional Neural Networks Educational Resource", padx= 10, pady= 20, packpadx= 10, packpady= 10)

        rootFrameRowCounter = 0

        baseFrame = self.objects.frame("This program allows you to design your very own Deep Convolutional Neural Networks that can learn to predict the genre of any music!", 
            screen= rootFrame, padx= 50, packpadx= 0, packpady= 0)

        dataFrame = self.objects.frame("Import Raw Data", grid= (rootFrameRowCounter, 0), screen= baseFrame)

        self.objects.label("                                                            ", 
            grid= (rootFrameRowCounter, 1), screen= baseFrame)

        jsonFrame = self.objects.frame("Import Proccessed Data", grid= (rootFrameRowCounter, 2), pady= 46, screen= baseFrame)
        rootFrameRowCounter+=1

        self.objects.oneLineGap((rootFrameRowCounter, 0), screen= baseFrame)
        rootFrameRowCounter+=1

        self.objects.label("                                                            ", 
            grid= (rootFrameRowCounter, 0), screen= baseFrame)

        CNNFrame = self.objects.frame("Deep Convolutional Neural Network", grid= (rootFrameRowCounter, 1), screen= baseFrame)

        #defining the objects

        #Importing Data Frames
        self.objects.button("Import Dataset", self.objects.importdDataSetPath, padx= 56, screen= dataFrame)

        self.objects.oneLineGap(screen= dataFrame)

        self.objects.button("Save Data After Processing", self.objects.importSavePath, padx= 19, screen= dataFrame)

        #Importing the json data
        self.objects.button("Import Processed data", self.objects.importJson, screen= jsonFrame)

        #Tuning the data processing and going to the CNN
        self.objects.button("Tune Data Processing\n *Pun Intended*", dataPreProcess, padx= 61, screen= CNNFrame)
        
        self.objects.oneLineGap(screen= CNNFrame)

        self.objects.button("Go to Convolutional Neural Network", self.CNN, screen= CNNFrame)

        self.objects.oneLineGap(screen= rootFrame)

        #exit and special notes
        self.objects.button("Exit", sys.exit, padx= 50, screen= rootFrame)

        self.objects.oneLineGap(screen= rootFrame)
        self.objects.oneLineGap(screen= rootFrame)
        self.objects.oneLineGap(screen= rootFrame)
        self.objects.label("////////////////////////////////////////////////////////////////////////////////////////////", screen= rootFrame)
        self.objects.label("This project was heavily inspired by tensorflow's playground, Harvard CS50's Introduction to Artifical Course and my best friend's and my love for music!", screen= rootFrame)
        self.objects.label("This is my first project towards one of my main goals of providing fun, interative and free educational reources for everyone!", screen= rootFrame)

    def CNN(self):
        self.objects.eraseAllObjects()
        buildCNN()


#indepented Sub Screen built on top of root based on root
class dataPreProcess():
    """
    This screen is to change the parameters of how the raw data is 
    proccessed and massaged into training data
    """
    def __init__(self):

        hyperParameterScreens(
            screenName= "Tune Data Processing", 
            fieldNames= ("Samples:", "Time:", "MFCCs:", "FFTs:", "HopLen:", "Segments:"), 
            defaultValues= (22050, 30, 40, 2048, 512, 10),
            fieldKeys= ("samples", "time", "mfccs", "ffts", "hopLen", "segments"),
            funcToBeCalled= dataPreProcess.setHyperParameters,
        )

    @staticmethod
    def setHyperParameters(fields, fieldKeys, screen):
        """
        Collect the data from the objects and update data tuning parameters accodingly
        by shooting their targets
        Action: Button Press
        Data shot: Parameters
        Target: properties of the dataSet object eg: self.samples(self == prepareData.data)
        """

        try:
            TuningHP = {}
            for index in range(len(fields)):
                TuningHP[fieldKeys[index]] = int(fields[index].get())

            dataSet.samples = TuningHP["samples"]
            dataSet.time = TuningHP["time"]
            dataSet.mfccs = TuningHP["mfccs"]
            dataSet.ffts = TuningHP["ffts"]
            dataSet.hopLen = TuningHP["hopLen"]
            dataSet.segments = TuningHP["segments"]

        except:
            err = errors()
            err.parametersIssue()

        screen.destroy()

"""
//////////////////
Building Screen
//////////////////
"""

class buildCNN():
    """
    This is where the user can build his/her own Deep Convolutional Network
    The user can add:
        i) Convolution Layers
        ii) Hidden Layers
        iii) New Training Parameters
        iv) Stop training when parameter does not decrease
    The user can then do one of three things:
        i) Use a default deep CNN written by me
        ii) Use his/her own amazing custom deep CNN
        iii) Import a previously saved model and use that
    """

    #initalize class variables
    objects = genericViewObjects(rootScreen.root)

    Convolutions = []
    HiddenLayers = [] 

    geometryCount = 5.15

    listToKeepTrack = [[], []]

    showModelFrame = None

    def __init__(self):
        """
        Initialize
        """

        self.show()
        
    def show(self):
        """
        Display all the immediate objects to be displayed to the user on initiation
        """

        self.rootFrame = self.objects.frame(padx= 0, pady= 0, packpadx = 20, packpady= 10)

        rootScreen.root.geometry("1000x" + str(int(self.geometryCount*100)))
        self.geometryCount+=1

        #defining the frames

        baseRowCounter = 0

        baseFrame = self.objects.frame("This is your Workstation, The only limit here is your imagination!", padx= 50, packpadx= 0, packpady= 0, screen= self.rootFrame)
        self.objects.oneLineGap()

        buildFrame = self.objects.frame("Build your own Deep Convolutional Neural Network", grid= (baseRowCounter, 0), screen= baseFrame)

        self.objects.label("                                                            ", 
            grid= (baseRowCounter, 1), screen= baseFrame)

        parametersFrame = self.objects.frame("Adjust Training Parameters", pady= 48, grid= (baseRowCounter, 2), screen= baseFrame)
        baseRowCounter+=1

        self.objects.oneLineGap((baseRowCounter, 0), screen= baseFrame)
        baseRowCounter+=1

        self.objects.label("                                                            ", 
            grid= (baseRowCounter, 0), screen= baseFrame)

        chooseModelFrame = self.objects.frame("Choose Model", grid= (baseRowCounter, 1), screen= baseFrame)
        baseRowCounter+=1

        #defining the objects

        #Build Frame
        self.objects.button("Add a Convolution and Pooling Layer", addConvolutionalLayer, screen= buildFrame)
        self.objects.oneLineGap(screen= buildFrame)

        self.objects.button("Add a Dense Hidden Layer", addHiddenLayer, padx= 51, screen= buildFrame)
        self.objects.oneLineGap(screen= buildFrame)

        self.objects.button("Remove the Top Layer", self.removeLine, padx= 64, screen= buildFrame)

        #Parameters Frame
        self.objects.button("Adjust Training Hyper Parameters", trainParameters, screen= parametersFrame)
        self.objects.oneLineGap(screen= parametersFrame)

        self.objects.button("Adjust Early Stopping", earlyStopping, padx= 57, screen= parametersFrame)
        
        #Model Frame
        self.objects.button("Train the Custom Model" , self.makeAndTrain, padx= 19, screen= chooseModelFrame)
        self.objects.oneLineGap(screen= chooseModelFrame)

        self.objects.button("Train the Default Model", self.useDefault, padx= 21, screen= chooseModelFrame)
        self.objects.oneLineGap(screen= chooseModelFrame)

        self.objects.button("Use Saved Model", self.useSavedModels, padx= 39, screen= chooseModelFrame)


    @classmethod
    def addLayer(cls, HyperParameters, layer = None):
        """
        This Class Method can be accessed outside of the class
        This methods allows the code to input convolution layers and Hidden layers
        It also displays them on the screen
        """
        #if frame does not exist, make it
        if cls.showModelFrame is None:
            cls.showModelFrame = cls.objects.frame("Your custom Deep Convolutional Neural Network", screen= rootScreen.root)
            cls.geometryCount+=1.3

        cls.geometryCount+=0.225
        rootScreen.root.geometry("1000x" + str(int(cls.geometryCount*100)))

        #checking which layer and doing actions accordingly
        if layer == "addConvolution":
            cls.Convolutions.append(HyperParameters)
            cls.listToKeepTrack[1].append(HyperParameters)

            cls.addLabel(cls, f"--Convolution Layer--   {HyperParameters}   --Convolution Layer--", cls.showModelFrame, True)

        elif layer == "addDense":
            cls.HiddenLayers.append(HyperParameters)
            cls.listToKeepTrack[1].append(HyperParameters)

            cls.addLabel(cls, f"--Hidden Dense Layer--   {HyperParameters}   --Hidden Dense Layer--", cls.showModelFrame, True)

        elif layer == "addTrainParameters":
            cls.geometryCount+=0.225
            rootScreen.root.geometry("1000x" + str(int(cls.geometryCount*100)))

            keys = list(HyperParameters.keys())
            text1 = {key : HyperParameters[key] for key in keys[:6]}
            text2 = {key : HyperParameters[key] for key in keys[6:]}

            cls.addLabel(cls, f"--New Train Parameters--    {text1}    --New Train Parameters-- \n --New Train Parameters--     {text2}      --New Train Parameters--", cls.showModelFrame)
            
        elif layer == "addEarlyStop":
            cls.addLabel(cls, f"--Added Early Stopping--   {HyperParameters}   --Added Early Stopping--", cls.showModelFrame)

        else:
            cls.addLabel(cls, str(HyperParameters))

    def addLabel(self, text, screen = None, addChoice= False):
        """
        Add a layer to the screen
        """

        label = self.objects.label(text, screen= screen)

        if addChoice:
            self.listToKeepTrack[0].append(label)

            
    def useDefault(self):
        """
        This is to use the default model given in the code i.e. 
        do not shoot actions at genreClassifier.convolutionLayers or genreClassifier.HiddenLayers targets
        """

        self.rootFrame.destroy()

        createAndTrainModels(model = "default")

    def makeAndTrain(self):
        """
        This is to make the custom model as specified by the user
        """

        self.rootFrame.destroy()

        createAndTrainModels(self.Convolutions, self.HiddenLayers, model = "custom")

    def useSavedModels(self):
        """
        This is to an old saved model instead of making a new one and training it.
        Saves a lot of time!
        """

        self.rootFrame.destroy()

        self.objects.importSavedModel()
        createAndTrainModels(model = "saved")     

    def removeLine(self):
        """
        This is to remove a layer of the user wants to do so
        If a layer exists, Try to remove it. 
        If it does not exist, display an error message
        """
        #Try removing the top layer
        try:
            label, layer = self.listToKeepTrack[0].pop(), self.listToKeepTrack[1].pop()
            
            if layer in self.Convolutions:
                self.Convolutions.remove(layer)
            elif layer in self.HiddenLayers:
                self.HiddenLayers.remove(layer)

            label.destroy()

        except:
            #Error screen 
            screen = genericViewObjects.newScreen("Error")
            screen.geometry("200x100")
            objects = genericViewObjects(screen)
            objects.label("No Layers Exist!")
            objects.button("<- Back", screen.destroy)


#Sub Screen built on top of root. In the context of the class buildCNN()
class addConvolutionalLayer():
    """
    This screen adds a single convolution and pooling layer to the custom Neural Network!
    """

    def __init__(self):
        """
        Intialize the data
        """ 

        hyperParameterScreens(
            screenName= "Convolution and Pool Hyper Parameters",
            fieldNames= ("No of Filters:", "Kernel:", "Activation:", "Pool Kernel:", "Pool Strides:", "Pool Padding:"),
            defaultValues= (32, str((3, 3)), "relu", str((3, 3)), str((2, 2)), "same"),
            fieldKeys= ("filters", "kernel", "activation", "pKernel", "pStride", "pPad"),
            funcToBeCalled= self.addConv,
        )

    @staticmethod
    def addConv(fields, fieldKeys, screen):
        """
        Add the hyperparameter inputs to the convolution layer list
        if they are legal, else, do nothing
        """

        try:
            convHP = {}
            for index in range(len(fields)):
                convHP[fieldKeys[index]] = (
                        int(fields[index].get()) if index == 0 else 
                        str(fields[index].get())
                )

            if convHP[fieldKeys[2]] not in activationFuncs or convHP[fieldKeys[-1]] not in padding:
                raise Exception
            
            buildCNN.addLayer(convHP, "addConvolution")

        except:
            err = errors()
            err.parametersIssue()

        screen.destroy()


#Sub Screen built on top of root. In the context of the class buildCNN()
class addHiddenLayer():
    """
    This screen adds a single Hidden layer to the custom Neural Network
    """

    def __init__(self):
        """
        Intialize the data
        """

        hyperParameterScreens(
            screenName= "Hidden Dense Layer Hyper Parameters",
            fieldNames= ("Neurons:", "Activation:", "Dropout:"),
            defaultValues= (128, "relu", 0.5), 
            fieldKeys= ("neurons", "activation", "dropout"),
            funcToBeCalled= self.addHidden
        )

    @staticmethod
    def addHidden(fields, fieldKeys, screen):
        """
        Add the hyperparameter inputs to the convolution layer list
        if they are legal, else, do nothing
        """

        try:

            LayerHP = {}
            for index in range(len(fields)):
                LayerHP[fieldKeys[index]] = (
                        str(fields[index].get()) if index == 1 else 
                        float(fields[index].get()) if index == 2 else 
                        int(fields[index].get())
                    )

            if LayerHP[fieldKeys[1]] not in activationFuncs:
                raise Exception

            buildCNN.addLayer(LayerHP, "addDense")
        except:

            err = errors()
            err.parametersIssue()

        screen.destroy()

#Sub Screen built on top of root. In the context of the class buildCNN()
class trainParameters():
    """
    This screen will allow the user to set his own method of training the deep CNN
    """
    
    def __init__(self):
        """
        Intialize the data
        """

        hyperParameterScreens(
            screenName= "Training Hyper Parameters",
            fieldNames= ("Test set:", "Validation set:", "Learning rate:", "Batch Size:", "Epochs:", "Verbose:", "Loss:", "Metrics:", "Early Stop:"),
            defaultValues= (0.2, 0.2, 0.001, 32, 15, 1, "sparse_categorical_crossentropy", 'accuracy', "True",),
            fieldKeys= ("test", "validate", "lr", "batch", "epochs", "verbose", "loss", "metrics","earlyStop"),
            funcToBeCalled= self.setTrainParam
        )

    @staticmethod
    def setTrainParam(fields, fieldKeys, screen):
        """
        Collect the data from the objects and update data tuning parameters accodingly
        by shooting their targets
        Action: Button Press
        Data shot: Parameters
        Target: properties of the genreClassifier object eg: self.samples(self == prepareData.data)
        """

        try:

            specs = {}
            for index in range(len(fields) -1):
                    specs[fieldKeys[index]] = (
                        float(fields[index].get()) if index in(0, 1, 2) 
                        else int(fields[index].get()) if index in (3, 4, 5)
                        else str(fields[index].get()) if index == 6
                        else list(str(fields[index].get()).split(","))
                    )

            specs["earlyStop"] = True if (fields[-1].get()) == "True" else False

            genreClassifier.test = specs['test']
            genreClassifier.validate = specs['validate']
            genreClassifier.lr = specs['lr']
            genreClassifier.batch = specs['batch']
            genreClassifier.epochs = specs['epochs']
            genreClassifier.verbose = specs['verbose']
            genreClassifier.loss = specs['loss']
            genreClassifier.metrics = specs['metrics']
            genreClassifier.earlyStop = specs['earlyStop']

            buildCNN.addLayer(specs, "addTrainParameters")

        except:
            err = errors()
            err.parametersIssue()

        screen.destroy()


#Sub Screen built on top of root. In the context of the class buildCNN()
class earlyStopping():
    """
    This screen will allow the user to set parameters for early stopping
    """

    def __init__(self):
        """
        Intialize the data
        """

        hyperParameterScreens(
            screenName= "Early Callback Parameters",
            fieldNames=  ("Monitor:", "Min Delta:", "Patience:", "Verbose:", "mode:"),
            defaultValues= ("val_acc", 0, 2, 1, "auto"),
            fieldKeys= ("monitor", "delta", "patience", "verbose", "mode"),
            funcToBeCalled= self.setEarlyStop
        )

    @staticmethod
    def setEarlyStop(fields, fieldKeys, screen):
        """
        Collect the data from the objects and update data tuning parameters accodingly
        by shooting their targets
        Action: Button Press
        Data shot: Parameters
        Target: properties of the genreClassifier object eg: self.samples(self == prepareData.data)
        """

        try:

            specs = {}
            for index in range(len(fields)):
                    specs[fieldKeys[index]] = (
                        int(fields[index].get()) if index in (1, 2, 3)
                        else str(fields[index].get())
                    )
                
            genreClassifier.monitor = specs['monitor']
            genreClassifier.delta = specs['delta']
            genreClassifier.patience = specs['patience']
            genreClassifier.callbackVerbose = specs['verbose']
            genreClassifier.mode = specs['mode']

            buildCNN.addLayer(specs, "addEarlyStop")

        except:
            err = errors()
            err.parametersIssue()

        screen.destroy()


"""
//////////////////
Testing Screen
//////////////////
"""

class createAndTrainModels():
    """
    Here is where the model gets compiled and trained
    wether it is custom, default or saved, everything happens here
    
    Once the model has finished training, The user can:
        i) see the testing accuracy and testing error 
        ii) Look at loss charts for the model
        iii) Save the model
        iv) Predict something using the model
        v) Play a guessing game with the model they just wrote to see who is superior!
    """

    objects = genericViewObjects(rootScreen.root)

    def __init__(self, convolutionLayers = None, HiddenLayers = None, model = "custom"):
        """
        Display all the immediate objects to be displayed to the user on initiation
        Call the main method from the model which complies and trains the model!
        """

        #intialize the root
        self.rootFrame = self.objects.frame(padx= 0, pady= 0, packpady= 10)
        rootScreen.root.geometry("1000x" + str(int((buildCNN.geometryCount- 1.1)*100)))
        
        frameRowCounter = 0
        
        #initlize the base and stats frame
        baseFrame = self.objects.frame("Test the Model", grid = (frameRowCounter, 0), screen= self.rootFrame)

        statsFrame = self.objects.frame("Model Statistics", grid= (frameRowCounter, 0), padx= 27, pady= 22, screen= baseFrame)
        self.objects.label("                             ", 
            grid= (frameRowCounter, 1), screen= baseFrame)

        """
        This block of code checks if the model passed is custom, saved or default.
        If it's custom then it updates the genreClassifier objects and describes the structure of the deep CNN to it
        Data shot: Structure of CNN
        Target: property of the genreClassifier object eg: self.samples(self == prepareData.data)
        If it's saved then load the model 
        If it is default then train the default model
        If it is default or custom then after trianing give the option to draw the loss graph for it
        Run the main of genreClassifier object in all three cases, genreClassifier.main
        """

        if model == "custom":
            
            chartFrame = self.objects.frame("Model Charts", padx= 70, pady= 32, grid= (frameRowCounter, 2), screen= baseFrame)
            frameRowCounter+=1

            self.allConv = convolutionLayers
            self.allHiddenLayers = HiddenLayers

            #If any inputs then shoot action at target of genreClassifier
            if len(self.allConv) > 0:
                genreClassifier.convolutionLayers = self.allConv
            if len(self.allHiddenLayers) > 0:
                genreClassifier.HiddenLayers = self.allHiddenLayers
        
            try:
                genreClassifier.main(dataSet)

            except:
                err = errors()
                if genreClassifier.error == "data":
                    err.dataIssue()
                elif genreClassifier.error == "CNN":
                    err.cnnIssue()
                else:
                    err.unknownIssue()
            
            self.objects.button("View Charts of the model", self.plotModel, padx= 37, screen= chartFrame)

        elif model == "saved":

            try:
                genreClassifier.main(dataSet)

            except:
                err = errors()
                if genreClassifier.error == "data":
                    err.dataIssue()
                elif genreClassifier.error == "CNN":
                    err.cnnIssue()
                else:
                    err.unknownIssue()

        elif model == "default":

            chartFrame = self.objects.frame("Model Charts", padx= 70, pady= 32, grid= (frameRowCounter, 2), screen= baseFrame)
            frameRowCounter+=1

            try:
                genreClassifier.main(dataSet)

            except:
                err = errors()
                if genreClassifier.error == "data":
                    err.dataIssue()
                elif genreClassifier.error == "CNN":
                    err.cnnIssue()
                else:
                    err.unknownIssue()
            
            self.objects.button("View Charts of the model", self.plotModel, padx = 37, screen= chartFrame)

        self.objects.oneLineGap(grid= (frameRowCounter, 0), screen= baseFrame)
        frameRowCounter+=1
        
        testFrame = self.objects.frame("Play around with the model", padx= 68, grid= (frameRowCounter, 0), screen= baseFrame)

        self.objects.label("                             ", 
            grid= (frameRowCounter, 1), screen= baseFrame)

        saveFrame = self.objects.frame("Save the Model", padx= 69, grid= (frameRowCounter, 2), screen= baseFrame)


        #Stats frame 
        self.objects.label(str(f"The accuracy of the model is: {genreClassifier.accuracy}"), screen= statsFrame)

        self.objects.label(str(f"The loss of the model is: {genreClassifier.modelError}"), screen= statsFrame)
        
        #Test Frame
        self.objects.button("Predict", self.predict, padx= 107, screen= testFrame)
        
        self.objects.oneLineGap(screen= testFrame)

        self.objects.button("Play a guessing game against the AI", Game, screen= testFrame)

        #Save Frame
        self.objects.button("Save Model", self.save, padx= 79, screen= saveFrame)

        self.objects.oneLineGap(screen= saveFrame)
        
        self.objects.button("Exit", sys.exit, padx= 103, screen= saveFrame)

    def save(self):
        """
        Saving the newly designed model
        """
        savePath = filedialog.asksaveasfilename()
        if savePath:
            genreClassifier.saveModel(savePath)
    
    #Sub Screen built on top of root. In the context of the class createAndTrainModels()
    def predict(self):
        """
        Predicting using the chosen model
        """

        self.objects.importPredictPath()
        self.AIprediction()

    def AIprediction(self):
        """
        Making a new screen and showing the prediction results there 
        """

        screen = genericViewObjects.newScreen("Prediction", "500x100")
        screenObjects = genericViewObjects(screen)
        
        screenObjects.label(f"The AI thinks that the genre of this song is : {genreClassifier.predict(dataSet)}")

        screenObjects.button("<- Back", screen.destroy)

    #Sub Screen built on top of root. In the context of the class createAndTrainModels()
    def plotModel(self):
        """
        Plotting the loss graphs for the model
        """
        #If it is possible to plot
        if genreClassifier.modelHistory is not None:

            #get the loss/accuracy history
            history = genreClassifier.modelHistory

            _, axis = plt.subplots(2)

            #plot the y axis
            axis[0].plot(history.history["acc"], label="train accuracy")
            axis[0].plot(history.history["val_acc"], label="test accuracy")
            axis[0].set_ylabel("Accuracy")
            axis[0].legend(loc="lower right")
            axis[0].set_title("Accuracy eval")
            
            #plot the x axis
            axis[1].plot(history.history["loss"], label="train error")
            axis[1].plot(history.history["val_loss"], label="test error")
            axis[1].set_ylabel("Error")
            axis[1].set_xlabel("Epoch")
            axis[1].legend(loc="upper right")
            axis[1].set_title("Error eval")

            #show the plot
            plt.show()

#Sub Screen built on top of root. In the context of the class createAndTrainModels()
class Game():

    def __init__(self):
        """
        Initialize
        """
        self.playGame()


    def playGame(self):
        """
        Make a new screen for the user to play a game against his/her own AI
        Intialize the introductory labels
        Intialize the buttons to play the music
        Intialize the Text Fields for the user to enter his/her predictions
        """

        #Start with defining the screeens
        self.gameScreen = genericViewObjects.newScreen("Music Genre Guess Game")
        self.objects = genericViewObjects(self.gameScreen)
        self.localRowCount = 0

        self.baseFrame = self.objects.frame("Guess the genre of each song!")

        self.inputFrame = self.objects.frame("Guess the Genre", grid= (0, 0), screen= self.baseFrame)

        musicFrame = self.objects.frame("Play Music", grid= (0, 1), screen= self.baseFrame)

        path = filedialog.askdirectory()
        
        if not path:
            self.gameScreen.destroy()
            return

        self.files, self.AIpredictions = genreClassifier.Game(path, 5, dataSet)

        #Initialize the objects
        self.buttons= []
        self.textFields = []

        #input and music frame, simultaneously 
        
        #Song1
        self.buttons.append(self.objects.button("Song 1", self.playMusic0, (self.localRowCount, 1), screen= musicFrame))
        self.textFields.append(self.objects.textField("", grid= (self.localRowCount, 0), screen= self.inputFrame))
        self.localRowCount+=1
        
        #Song2
        self.buttons.append(self.objects.button("Song 2", self.playMusic1, (self.localRowCount, 1), screen= musicFrame))
        self.textFields.append(self.objects.textField("", grid= (self.localRowCount, 0), screen= self.inputFrame))
        self.localRowCount+=1

        #Song3
        self.buttons.append(self.objects.button("Song 3", self.playMusic2, (self.localRowCount, 1), screen= musicFrame))
        self.textFields.append(self.objects.textField("", grid= (self.localRowCount, 0), screen= self.inputFrame))
        self.localRowCount+=1
        
        #Song4
        self.buttons.append(self.objects.button("Song 4", self.playMusic3, (self.localRowCount, 1), screen= musicFrame))
        self.textFields.append(self.objects.textField("", grid= (self.localRowCount, 0), screen= self.inputFrame))
        self.localRowCount+=1

        #Song5
        self.buttons.append(self.objects.button("Song 5", self.playMusic4, (self.localRowCount, 1), screen= musicFrame))
        self.textFields.append(self.objects.textField("", grid= (self.localRowCount, 0), screen= self.inputFrame))
        self.localRowCount+=1

        #A gap of line
        self.objects.oneLineGap( (self.localRowCount, 0), screen= self.baseFrame)
        self.localRowCount+=1

        #results and exit buttons, base frame
        self.resultsButton = self.objects.button("Get Results", self.showResults, (self.localRowCount, 0), screen= self.baseFrame)

        self.objects.button("Exit", self.gameScreen.destroy, (self.localRowCount, 1), screen= self.baseFrame)
        self.localRowCount+=1


    def playMusic0(self):
        """
        Play music
        """
        
        #play music using pygame 
        pygame.mixer.init()
        sound = pygame.mixer.Sound(self.files[0])
        sound.play()
        time.sleep(30)
        sound.stop()

    def playMusic1(self):
        """
        Play music
        """

        #play music using pygame 
        pygame.mixer.init()
        sound = pygame.mixer.Sound(self.files[1])
        sound.play()
        time.sleep(30)
        sound.stop()
        

    def playMusic2(self):
        """
        Play music
        """

        #play music using pygame 
        pygame.mixer.init()
        sound = pygame.mixer.Sound(self.files[2])
        sound.play()
        time.sleep(30)
        sound.stop()
    
    def playMusic3(self):
        """
        Play music
        """

        #play music using pygame 
        pygame.mixer.init()
        sound = pygame.mixer.Sound(self.files[3])
        sound.play()
        time.sleep(30)
        sound.stop()
    
    def playMusic4(self):
        """
        Play music
        """

        #play music using pygame 
        pygame.mixer.init()
        sound = pygame.mixer.Sound(self.files[4])
        sound.play()
        time.sleep(30)
        sound.stop()

    def showResults(self):
        """
        Compare the user's predictions against his own AI's predictions
        """

        #Start with getting the list of the user predictions and AI predictions
        checkList = [[], []]
        checkList[0] = [human.lower() for human in [str(textField.get()) for textField in self.textFields]]
        checkList[1] = [i for i in self.AIpredictions]

        #Destroy the input Frame as we will later replace this with the results frame
        self.inputFrame.destroy()

        #Strings to print
        str1 = "You and the AI had the same prediction!"

        str2 = "You and the AI had different predictions :/"

        #set the results frame
        self.resultFrame = self.objects.frame("The Results", grid= (0, 0), screen= self.baseFrame)

        #print out the results
        for i in range(5):
            self.objects.label(
            f"For song {i}, Your prediction was: {[checkList[0][i]]} and your AI's prediction was {checkList[1][i]}\n {str1 if checkList[0][i] == checkList[1][i] else str2}",
            (self.localRowCount, 0), screen= self.resultFrame
            )
            self.localRowCount+=1   

        #destroy the results button 
        self.resultsButton.destroy()


"""
Initialize the root screen and loop it
"""       

#initialize root screen
rootScreen()

#Main looping
mainloop()