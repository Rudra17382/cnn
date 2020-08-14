import numpy as np 
import pathlib, os, json, random
from sklearn.model_selection import train_test_split 
import tensorflow as tf
import prepareData

activationFuncs = ["deserialize", "elu", "exponential", "get", "hard_sigmoid", "linear", 
"relu", "selu", "serialize", "sigmoid", "softmax", "softplus", "softsign", "swish", "tanh"
]
padding = ["valid", "same"]

keras = tf.keras
layer = keras.layers

class CNN():

    def __init__(self):
        """
        Initalizing Targets and properties
        """

        #start with model
        self.model = None 
        #start with saved model?
        self.savedModel = None

        #error and accuracy
        self.modelError = None
        self.accuracy = None 

        #all the data
        self.mfcc = None
        self.label = None
        self.map = None

        #convolution and hidden layers *for custom networks*
        self.convolutionLayers = []
        self.HiddenLayers = []
        
        #save history
        self.modelHistory = None
        
        #set a default predict path
        self.predictPath = None

        #set default test and validate proportions
        self.test = 0.2
        self.validate = 0.2
        #set other training parameters for model.fit
        self.lr = 0.001
        self.batch = 32
        self.epochs = 15
        self.verbose = 1
        
        #using callback to stop early upon condition example: validation loss goes up
        self.earlyStop = True

        #set callback parameters
        self.monitor = 'val_acc'
        self.delta = 0
        self.patience = 2
        self.callbackVerbose = 0
        self.mode = 'auto'

        #compile parameters
        self.loss = "sparse_categorical_crossentropy"
        self.metrics = ['accuracy']

        self.error = None


    def main(self, dataObject):
        """
        create train-validate-test splits
        create the model and compile it. Fit the model
        set the resulting error and accuracy as targets for the GUI
        """

        try:
            #train, validate, test split
            xTrain, xValidate, xTest, yTrain, yValidate, yTest = self.prepare(self.test, self.validate, dataObject)

            #shape of the input data that is going to be fed to the model
            inputShape = (xTrain.shape[i] for i in range(1, 4))

        except:
            
            self.error = "data"
        
        try:
            #use saved model if asked, else make new model and train it 
            if self.savedModel is None:
                #make the model according to the input shape
                model = self.makeModel(inputShape)

                #compile the model 
                model.compile(
                    optimizer= keras.optimizers.Adam(lr= self.lr),
                    loss= self.loss,
                    metrics= self.metrics
                )

                #if use of callback is asked
                if self.earlyStop:

                    #create a early stopping callback
                    callback = keras.callbacks.EarlyStopping(
                            monitor= str(self.monitor), min_delta= int(self.delta), 
                            patience= int(self.patience), verbose= int(self.callbackVerbose), mode= str(self.mode)
                        )

                    #fit the model on the training data we computed earlier and save its history to plot it later
                    self.modelHistory = model.fit(xTrain, yTrain, validation_data= (xValidate, yValidate), 
                        callbacks=[callback], batch_size= self.batch, epochs= self.epochs, 
                        verbose= self.verbose)
                
                #if no early stopping callback then train the model normally
                else:
                    self.modelHistory = model.fit(xTrain, yTrain, validation_data= (xValidate, yValidate), 
                        batch_size= self.batch, epochs= self.epochs, verbose= self.verbose)
            else:
                #load in a saved model
                model = keras.models.load_model(self.savedModel)

            #get error and accuracy on the test set
            self.modelError, self.accuracy = model.evaluate(xTest, yTest, verbose= self.verbose)
            
            #get the summary for the model
            model.summary()

            #set the model property as model object
            self.model = model

        except:
            self.error = "CNN"
            
    
    def prepare(self, test, validate, dataObject):
        '''
        prepare and massage the data
        create (train, validate), test split
        create train, validate split 
        return train, validate, test split
        '''

        #get the data object and load data from it
        data = dataObject
        
        #make data properties
        self.mfcc, self.label, self.map = data.loadData()

        #use train test split to first split it into train and test, use train to split again into train and validate
        xTrain, xTest, yTrain, yTest = train_test_split(self.mfcc, self.label, test_size= test)
        xTrain, xValidate, yTrain, yValidate = train_test_split(xTrain, yTrain, test_size= validate)

        #add a new dimention to match the expected shape
        xTrain = xTrain[..., np.newaxis]
        xValidate = xValidate[..., np.newaxis]
        xTest = xTest[..., np.newaxis]

        return xTrain, xValidate, xTest, yTrain, yValidate, yTest

    def makeModel(self, inputShape):
        '''
        Make a sequential model, use given info and fill the gaps with defualt info
        assume targets on model have already been shot by the view controller
        i.e. model parameters have already been given by the GUI
        convert hyperParameters to convolution and Hidden Dense Layers and use them
        Return the proper model after considering all edge cases
        '''

        #start by getting hyperParameters recived on target properties
        Clayer, Hlayer, dropout = self.hyperParametersToLayers(inputShape)
        dropout = dropout if dropout != 0 else 0.5

        #add dropout and ouput layer if hidden layers specified by the user
        if len(Hlayer) > 1:
            Hlayer.append(layer.Dropout(dropout))
            Hlayer.append(layer.Dense(10, activation='softmax'))
        
        #define default set of layers

        #Default Convolutional and pooling layers
        defaultConvolutionLayers = [
            layer.Conv2D(64, (3, 3), activation= 'relu', input_shape= inputShape),
            layer.MaxPool2D((3, 3), strides= (2, 2), padding= 'same'),
            layer.BatchNormalization(),

            layer.Conv2D(64, (3, 3), activation= 'relu', input_shape= inputShape),
            layer.MaxPool2D((3, 3), strides= (2, 2), padding= 'same'),
            layer.BatchNormalization(),

            layer.Conv2D(32, (2, 2), activation= 'relu', input_shape= inputShape),
            layer.MaxPool2D((2, 2), strides= (2, 2), padding= 'same'),
            layer.BatchNormalization()
        ]

        #Default Hidden Layers
        defaultHiddenLayers = [
            layer.Flatten(),
            layer.Dense(128, activation= 'relu'),

            layer.Dropout(0.5),

            layer.Dense(10, activation='softmax')
        ]

        #According to what is missing, supplment it to make sure that a proper deep CNN exists
        if len(Clayer) == 0 and len(Hlayer) > 1:
            layers = defaultConvolutionLayers + Hlayer

        elif len(Hlayer) <= 1 and len(Clayer) != 0:
            layers = Clayer + defaultHiddenLayers

        elif len(Hlayer) <= 1 and len(Clayer) == 0:
            layers = defaultConvolutionLayers + defaultHiddenLayers

        else:
            layers = Clayer + Hlayer
        
        #return sequential model
        return keras.models.Sequential(layers)

    def predict(self, dataObject, testPath= None):
        """
        Predict given music's label by listening to it
        i.e. converting it into its mfccs and feeding it into our deep CNN
        """

        model = self.model

        #checking for test path, if it doenst exist, use the target property
        if testPath is not None:
            x = np.array(dataObject.oneClipMFCCs(testPath))
        else:
            x = np.array(dataObject.oneClipMFCCs(self.predictPath))
        
        # we need an array of shape (1, 130, 40, 1)
        x = x[..., np.newaxis]
        x = x[0]
        x = x[np.newaxis, ...]
        
        #predict on input
        prediction = model.predict(x)

        #take the max probability of the output
        predicted_index = np.argmax(prediction, axis=1)

        #return the map to the label
        return self.map[predicted_index[0]]


    def hyperParametersToLayers(self, inputShape):
        """
        Synethize layers based on the shots received by targets hanged at self by
        self.convLayers and self.HiddenLayers
        """

        #intialize some local storage objects
        convolution = []
        hidden = [layer.Flatten()]
        dropout = 0

        #for each layer in convolution layers
        for Clayer in self.convolutionLayers:
            try:
                
                #parsing the Hyper parameters
                filters = int(Clayer['filters'])
                kernel = self.stringToTuple(Clayer['kernel'])
                activation = str(Clayer['activation']) if Clayer['activation'] in activationFuncs else None
                pKernel = self.stringToTuple(Clayer['pKernel'])
                pStride = self.stringToTuple(Clayer['pStride'])
                pPad = str(Clayer['pPad']) if Clayer['pPad'] in padding else None

                #Add a Convolution layer with the parameters just parsed
                convolution.append(
                    layer.Conv2D(filters, kernel, activation= activation, input_shape= inputShape)
                )
                convolution.append(layer.MaxPool2D(pKernel, strides= pStride, padding= pPad))
                convolution.append(layer.BatchNormalization())
            except:
                #dont consider the layer as it is not complete/illegal
                
                self.error = "parameters"

        for HLayer in self.HiddenLayers:
            try:
                
                #parsing Hyper Parameters
                neurons = int(HLayer['neurons'])
                activation = str(HLayer['activation']) if HLayer['activation'] in activationFuncs else None
                dropout = int(HLayer['dropout']) if 0 <= int(HLayer['dropout']) <= 1 else 0
                
                #Add a Hidden layer with the parameters just parsed
                hidden.append(layer.Dense(neurons, activation= activation))
            except:
                #dont consider the layer as it is not complete/illegal

                self.error = "parameters"

        return convolution, hidden, dropout

    def stringToTuple(self, string):
        """
        Converting string to tuple
        """

        listForm = list(string)
        thingsToRemove = ["(", ")", ",", " "]
        return tuple(int(i) for i in listForm if i not in thingsToRemove)
        
    def saveModel(self, savePath):
        """
        Saving model
        """

        self.model.save(savePath)

    def Game(self, path, songsToPredict, dataObject):
        """
        Making predictions on random files for the game
        """

        #start by getting all the files in the directory
        onlyfiles = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

        #make an array of random files from the directory
        randomIndexes, count= [], 0
        while count < songsToPredict:
            randomNumber = random.randrange(0, len(onlyfiles))
            if randomNumber not in randomIndexes:
                randomIndexes.append(randomNumber)
                count+=1
        files = [onlyfiles[randomIndexes[i]] for i in range(songsToPredict)]

        #Make a prediction on each file using self.predict
        predictions = [self.predict(dataObject, testPath= files[i]) for i in range(songsToPredict)]

        return files, predictions