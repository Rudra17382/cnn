from tkinter import Tk, Toplevel, Button, Label, Entry, filedialog, mainloop, LabelFrame
import prepareData, genre

dataSet = prepareData.data()
genreClassifier = genre.CNN()


class genericViewObjects():
    """
    This is a class for generic view objects 
    it is to make it easier to code and manage so that you dont need to call 
    so many lines again and again for simple objects, instead just call one line

    Note: All objects will be made on the same screen which is it's propery
    """

    def __init__(self, screen):
        """
        init properties
        """

        self.screen = screen
        self.framesOnScreen = []
        self.labelsOnScreen = []
        self.textFieldsOnScreen = []
        self.buttonsOnScreen = []
    
    @staticmethod
    def newScreen(text= None, geometery= None):
        """
        Make a new screen Object
        """

        screen = Toplevel()

        if text is not None:
            screen.title(text)
        if geometery is not None:
            screen.geometry(geometery)

        return screen

    def label(self, text, grid= None, screen= None):
        """
        Make a new Label
        """

        label = Label(self.screen if screen is None else screen, text= text)

        if grid is not None:
            label.grid(row= grid[0], column= grid[1])
        else:
            label.pack()

        self.labelsOnScreen.append(label)
        return label
    
    def textField(self, default, width = 35, border = 3, grid = None, screen= None):
        """
        Make a new Text Field
        """ 

        textField = Entry(self.screen if screen is None else screen, width = width, borderwidth = border)
        textField.insert(0, default)

        if grid is not None:
            textField.grid(row= grid[0], column= grid[1])
        else:
            textField.pack()

        self.textFieldsOnScreen.append(textField)
        return textField

    def button(self, text, funcToBeCalled, grid = None, padx = 20, pady = 5, screen= None):
        """
        Make a new button
        """

        button = Button(self.screen if screen is None else screen, text= text, command= funcToBeCalled, padx= padx, pady= pady)

        if grid is not None:
            button.grid(row= grid[0], column= grid[1])
        else:
            button.pack()

        self.buttonsOnScreen.append(button)
        return button
    
    def frame(self, text= "", padx= 20, pady= 20, packpadx= 20, packpady= 20, grid= None, screen= None):
        """
        Make a new frame inide the current screen
        """ 

        frame = LabelFrame(self.screen if screen is None else screen, text= text, padx= padx, pady= pady)

        if grid is not None:
            frame.grid(row= grid[0], column= grid[1])
        else:
            frame.pack(padx= packpadx, pady= packpady)

        self.framesOnScreen.append(frame)
        return frame

    def eraseLabels(self):
        """
        Destroy all the labels from the current screen
        """
        
        for label in self.labelsOnScreen:
            label.destroy()
        self.labelsOnScreen = []

    def eraseTextFields(self):
        """
        Destroy all the text fields from the current screen
        """

        for textField in self.textFieldsOnScreen:
            textField.destroy()
        self.textFieldsOnScreen = []
    
    def eraseButtons(self):
        """
        Destroy all the buttons from the current screen
        """
        
        for button in self.buttonsOnScreen:
            button.destroy()
        self.buttonsOnScreen = []
    
    def eraseFrames(self):
        """
        Destroy all the frames from the current screen
        """

        for frame in self.framesOnScreen:
            frame.destroy()
        self.framesOnScreen = []
    
    def eraseAllObjects(self):
        """
        Destroy all Objects on the current screen excluding the current screen
        """
        
        self.eraseLabels()
        self.eraseTextFields()
        self.eraseButtons()
        self.eraseFrames()
    
    def importdDataSetPath(self):
        """
        Set the dataset path i.e. where to import dataset from if metadata file does not exist
        """
        
        dataPath = filedialog.askdirectory()
        if dataPath:
            dataSet.setNewDataPath = True
            dataSet.dataPath = dataPath
        else:
            err = errors()
            err.dataIssue()

    def importJson(self):
        """
        Set the metadata file path i.e. where to import data from
        """
        
        jsonPath = filedialog.askopenfilename()
        if jsonPath:
            dataSet.setNewJsonPath = True
            dataSet.jsonPath = jsonPath
        else:
            err = errors()
            err.dataIssue()

    def importSavePath(self):
        """
        Set Training Data path i.e. After processing data, where to save it.
        """

        savePath =  filedialog.asksaveasfilename()
        if savePath:
            dataSet.saveTrainingData = savePath
        else:
            err = errors()
            err.dataIssue()

    def importPredictPath(self):
        """
        Set prediction path i.e. where is the data to apply prediction on
        """
        
        predictPath = filedialog.askopenfilename()
        if predictPath:
            genreClassifier.predictPath = predictPath
        else:
            err = errors()
            err.improperPath()

    def importSavedModel(self):
        """
        Import a saved model and use it instead of making a new one.
        """
        
        savedModel = filedialog.askopenfilename()
        if savedModel:
            genreClassifier.savedModel = savedModel
        else:
            err = errors()
            err.improperPath()
    
    def oneLineGap(self, grid = None, screen= None):
        """
        Simple method to make one line gaps
        """
        self.label("", grid, screen)

class errors():

    def __init__(self):
        self.screen = genericViewObjects.newScreen("Error")
        self.screen.geometry("400x150")
        self.objects = genericViewObjects(self.screen)

    def parametersIssue(self):
        self.objects.label("Illegal Parameters Error")
        self.objects.label("You have entered illegal parameters! \n Please enter proper parameters")

        self.objects.button("<- Back", self.screen.destroy)


    def cnnIssue(self):
        self.objects.label("Tensorflow was not able to compile/train the network :/", (0, 0))

        self.objects.label("Please check the following and try again:", (1, 0))
        self.objects.label("\t1.Make sure that you have not put in too many convoutions\n This is because each convolution significantly shirnks the image.", 
            (2, 0))

        self.objects.button("<- Back", self.screen.destroy)
    
    def dataIssue(self):
        self.objects.label("Improper Data Error")
        self.objects.label("Make sure you are using a proper dataset \n and proper processed data")

        self.objects.button("<- Back", self.screen.destroy)

    def unknownIssue(self):
        self.objects.label("Unknown Error :/")

        self.objects.button("<- Back", self.screen.destroy)

    def improperPath(self):
        self.objects.label("Please enter a proper predict path")

        self.objects.button("<- Back", self.screen.destroy)

class hyperParameterScreens():

    def __init__(self, screenName, fieldNames, defaultValues, fieldKeys, funcToBeCalled):
        self.fields = []
        self.fieldNames = fieldNames
        self.defaultValues = defaultValues
        self.fieldKeys = fieldKeys
        self.funcToBeCalled = funcToBeCalled

        self.screen = genericViewObjects.newScreen(screenName)
        self.objects = genericViewObjects(self.screen)
        
        self.show()

    def show(self):
        """
        Intialize and show the objects
        """

        self.localRowCounter = 0
        for _, fieldName, default in zip(range(len(self.fieldNames)), self.fieldNames, self.defaultValues):
            self.objects.label(fieldName, (self.localRowCounter, 0))
            self.fields.append(self.objects.textField(default , grid=(self.localRowCounter, 1)))
            self.localRowCounter+=1

        self.objects.oneLineGap((self.localRowCounter, 0))
        self.localRowCounter+=1

        self.objects.label("                             ", (self.localRowCounter, 0))

        self.objects.button("Confirm Hyper Parameters", self.callFunc, (self.localRowCounter, 1))

        self.localRowCounter+=1

        self.objects.oneLineGap((self.localRowCounter, 0))
    
    def callFunc(self):
        self.funcToBeCalled(
            fields= self.fields,
            fieldKeys= self.fieldKeys,
            screen= self.screen
        )