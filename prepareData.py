import os, sys, pathlib, json
import math, numpy as np
import librosa

class data():

    def __init__(self):

        #initiate default Hyper Parameters
        self.samples = 22050
        self.time = 30

        self.mfccs = 40
        self.ffts = 2048
        self.hopLen = 512
        self.segments = 10
        
        #default file paths
        self.setNewDataPath = False
        self.dataPath = str(pathlib.Path(__file__).parent.absolute()) + os.sep + "dataset"
        self.setNewJsonPath = False
        self.jsonPath = str(pathlib.Path(__file__).parent.absolute()) + os.sep + "metadata"

        self.saveTrainingData = None

    def createMFCCs(self, datasetPath):
        """
        Create MFCCs and return them.
        use the directory names as genre labels for each song.
        datasetPath -> childDirectory -> file
                        (label)          (input)
        example:     dataset-> blues -> blues.0001.wav 

        data is collected in the form of a dictionary so 
        that it is easy to convert it later to json
        """ 
        
        #data to be collected
        data = {'map' : [], 'MFCC' : [], 'label' : []}  
        
        #get mfccs per segment
        samplesPerSegment = (self.samples * self.time) // self.segments   
        MFCCsPerSegment = math.ceil(samplesPerSegment / self.hopLen) 

        print("\n\n Generating MFCCs....", end= '')
        
        #for everything in given directory
        for label, (root, _, fileNames) in enumerate(os.walk(datasetPath)): 
            #if not looking at the root directory
            if root is not datasetPath:   
                
                #for all files, load it using librosa
                for currentFile in fileNames:   

                    #load the file with its signals and the sample rate 
                    filePath = os.path.join(root, currentFile)  
                    signal, sampleRate = librosa.load(filePath, sr= self.samples)
                    print(filePath)  

                    #for each segment, make an mfcc. Append mfcc to input data and genre label to label
                    for segment in range(self.segments): 
                        
                        #define the start and the end of a segment
                        start = samplesPerSegment * segment 
                        end = start + samplesPerSegment

                        #Extract MFCCs of the Segment
                        MFCC = librosa.feature.mfcc(signal[start:end],  
                                                    sampleRate,     
                                                    n_mfcc= self.mfccs,  
                                                    n_fft= self.ffts,     
                                                    hop_length= self.hopLen) 
                        MFCC = MFCC.T   
                        
                        #if the number of mfccs is what we expect, then write it to the data dict.
                        if len(MFCC) == MFCCsPerSegment:    
                            data['MFCC'].append(MFCC.tolist())                  
                            data['label'].append(label-1)  

                #map label to the current folder name which is the genre 
                currentLabel = root.split(os.sep)[-1]   
                data['map'].append(currentLabel) 

        print("\n\n Finished generating MFCCs! \n\n")

        #store property and return data
        self.map = data['map']
        return data 

    def loadData(self):
        """
        loading the data. 
        if new dataset path is giiven then use that to create new data and save it if specified
        else open json file if possible otherwise open the dataset and make the data, agaian, save it if specified
        """

        print(self.jsonPath, self.dataPath)

        #checking for the first condition
        if self.setNewDataPath and not self.setNewJsonPath:

            #try getting the data through dataset, if no data arrives then print error. if save path given, save it
            try:
                #if data is asked to be saved
                if self.saveTrainingData is not None:

                    #generate the data
                    data = self.createMFCCs(self.dataPath)
                    
                    #Write it to a json file
                    with open(self.saveTrainingData + ".json", "w") as jsonFile:
                        json.dump(data, jsonFile, indent= 4)

                else:
                    #data is not asked to be saved therfore just extract and use it without saving
                    data = self.createMFCCs(self.dataPath)

            except:
                #if not proper files then print error
                print("Error")

        else:
            #try getting the data through saved json. if not possible, make new data and save it if save path given

            #trying to fetch data from a json file
            try:
                with open(self.jsonPath + ".json", "r") as jsonFile:
                    data = json.load(jsonFile)

            except:

                #Go for extracting the data from the defauly dataset, if asked to be saved:
                if self.saveTrainingData is not None:

                    #Extract data
                    data = self.createMFCCs(self.dataPath)

                    #Write it to a json file 
                    with open(self.saveTrainingData + os.sep + ".json", "w") as jsonFile:
                        json.dump(data, jsonFile, indent= 4)

                else:
                    #data is not asked to be saved therfore just extract and use it without saving
                    data = self.createMFCCs(self.dataPath)

        #converting the first two to numpy arrays
        mfcc, label, maps = np.array(data['MFCC']), np.array(data['label']), data['map']

        #no data exists if we get no mfccs
        if len(mfcc) == 0:
            sys.exit("No training data exists or can be created!")

        return mfcc, label, maps
    
    def oneClipMFCCs(self, filePath):
        """
        Same procedure as data/createMFCCs except do it only for one file rather than an entire directory
        """  

        #data to be collected
        data = []

        #get mfccs per segment
        samplesPerSegment = (self.samples * self.time) // self.segments   
        MFCCsPerSegment = math.ceil(samplesPerSegment / self.hopLen) 

        print("\n\n Generating MFCCs for the prediction....", end= '')
        
        #check for correct format as librosa only works for .wav files
        if filePath.lower().endswith('.wav'):

            #load the file with its signals and the sample rate 
            signal, sampleRate = librosa.load(filePath, sr= self.samples)

            #for each segment, make an mfcc. Append mfcc to input data and genre label to label
            for segment in range(self.segments): 

                #define the start and the end of a segment
                start = samplesPerSegment * segment 
                end = start + samplesPerSegment

                #Extract MFCCs of the Segment
                MFCC = librosa.feature.mfcc(signal[start:end],  
                                            sampleRate,     
                                            n_mfcc= self.mfccs,  
                                            n_fft= self.ffts,     
                                            hop_length= self.hopLen) 
                MFCC = MFCC.T   

                #if the number of mfccs is what we expect, then write it to the data dict.
                if len(MFCC) == MFCCsPerSegment:    
                    data.append(MFCC.tolist())  
        
        print("\n\n Finished Generating MFCCs!", end= '')

        #return the data
        return data