from __future__ import print_function, absolute_import
import random
from collections import OrderedDict
from numba import njit
from numba import int32, deferred_type, optional, types
from numba import jitclass
from numba.typed import List
from numba import njit, typeof, typed, types
import numpy as np
import numba as nb
import cv2
from matplotlib import gridspec
from matplotlib import pyplot as plt
import pickle

def read_single_keypress():
    """Waits for a single keypress on stdin.

    This is a silly function to call if you need to do it a lot because it has
    to store stdin's current setup, setup stdin for reading single keystrokes
    then read the single keystroke then revert stdin back after reading the
    keystroke.

    Returns a tuple of characters of the key that was pressed - on Linux, 
    pressing keys like up arrow results in a sequence of characters. Returns 
    ('\x03',) on KeyboardInterrupt which can happen when a signal gets
    handled.

    """
    import termios, fcntl, sys, os
    fd = sys.stdin.fileno()
    # save old state
    flags_save = fcntl.fcntl(fd, fcntl.F_GETFL)
    attrs_save = termios.tcgetattr(fd)
    # make raw - the way to do this comes from the termios(3) man page.
    attrs = list(attrs_save) # copy the stored version to update
    # iflag
    attrs[0] &= ~(termios.IGNBRK | termios.BRKINT | termios.PARMRK
                  | termios.ISTRIP | termios.INLCR | termios. IGNCR
                  | termios.ICRNL | termios.IXON )
    # oflag
    attrs[1] &= ~termios.OPOST
    # cflag
    attrs[2] &= ~(termios.CSIZE | termios. PARENB)
    attrs[2] |= termios.CS8
    # lflag
    attrs[3] &= ~(termios.ECHONL | termios.ECHO | termios.ICANON
                  | termios.ISIG | termios.IEXTEN)
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    # turn off non-blocking
    fcntl.fcntl(fd, fcntl.F_SETFL, flags_save & ~os.O_NONBLOCK)
    # read a single keystroke
    ret = []
    try:
        ret.append(sys.stdin.read(1)) # returns a single character
        fcntl.fcntl(fd, fcntl.F_SETFL, flags_save | os.O_NONBLOCK)
        c = sys.stdin.read(1) # returns a single character
        while len(c) > 0:
            ret.append(c)
            c = sys.stdin.read(1)
    except KeyboardInterrupt:
        ret.append('\x03')
    finally:
        # restore old state
        termios.tcsetattr(fd, termios.TCSAFLUSH, attrs_save)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags_save)
    return tuple(ret)

model = None
with open('visionModel.pickle', 'rb') as handle:
    model = pickle.load(handle)

@njit
def hammingVector(stack1, stack2):
    return (stack1 != stack2).sum(axis=1)

#load data into JIT class
@njit
def compareLevelData(levelData, vec, metric=hammingVector):
    lenLevelData = levelData.shape[0]
    print(levelData, levelData.shape)
    lenFeature = levelData.shape[1]
    testColumn = np.zeros(shape=(lenLevelData, lenFeature), dtype=levelData.dtype)
    for i in range(0, lenLevelData):
        testColumn[i] = vec
        pairwiseDistance =  metric(levelData, vec)
        minIndex = np.argmin(pairwiseDistance)
    return np.argmin(pairwiseDistance)

class VisionModelPacker:
    def __init__(self, filename=None):
        if filename is not None:
            self.loadModel(filename)
            self.packVisionModelEvaluator()
        else:
            self.fileModel = None
    def loadModel(self, filename):
        with open('visionModel.pickle', 'rb') as handle:
            self.fileModel = pickle.load(handle)
    def packVisionModelEvaluator(self, featureExample = [[1],[1]]):
        if self.fileModel is None:
            print("Please load a model first")
        else:
            #define compute model
            self.computeModel = {} 
            self.computeModel["wordWeight"] = typed.Dict.empty(
                key_type=types.string,
                value_type=types.float64,
            )
            self.computeModel["wordIndex"] = typed.Dict.empty(
                key_type=types.string,
                value_type=types.int64,
            )
            featureExample2 = np.asarray(featureExample)
            typeofFeatureExample2 = typeof(featureExample2)
            self.computeModel["data"] = typed.Dict.empty(
                key_type=types.string,
                value_type=typeofFeatureExample2,
            )
            self.computeModel["children"] = typed.Dict.empty(
                key_type=types.string,
                value_type=nb.types.ListType(types.string),
            )
            
            self.computeModel["intUnicodeMap"] = typed.Dict.empty(
                key_type=types.int64,
                value_type=types.string,
            )
            
            self.computeModel["intUnicodeMap"][0] = "0"
            self.computeModel["intUnicodeMap"][1] = "1"
            self.computeModel["intUnicodeMap"][2] = "2"
            self.computeModel["intUnicodeMap"][3] = "3"
            self.computeModel["intUnicodeMap"][4] = "4"
            self.computeModel["intUnicodeMap"][5] = "5"
            self.computeModel["intUnicodeMap"][6] = "6"
            self.computeModel["intUnicodeMap"][7] = "7"
            self.computeModel["intUnicodeMap"][8] = "8"
            self.computeModel["intUnicodeMap"][9] = "9"
            self.computeModel["intUnicodeMap"][10] = "10"
            self.computeModel["intUnicodeMap"][11] = "11"
            self.computeModel["intUnicodeMap"][12] = "12"
            self.computeModel["intUnicodeMap"][13] = "13"
            self.computeModel["intUnicodeMap"][14] = "14"
            self.computeModel["intUnicodeMap"][15] = "15"
            self.computeModel["intUnicodeMap"][16] = "16"
            self.computeModel["intUnicodeMap"][17] = "17"
            self.computeModel["intUnicodeMap"][18] = "18"
            self.computeModel["intUnicodeMap"][19] = "19"
            self.computeModel["intUnicodeMap"][20] = "20"
            self.computeModel["intUnicodeMap"][21] = "21"
            self.computeModel["intUnicodeMap"][22] = "22"
            self.computeModel["intUnicodeMap"][23] = "23"
            self.computeModel["intUnicodeMap"][24] = "24"
            self.computeModel["intUnicodeMap"][25] = "25"
            self.computeModel["intUnicodeMap"][26] = "26"
            self.computeModel["intUnicodeMap"][27] = "27"
            self.computeModel["intUnicodeMap"][28] = "28"
            self.computeModel["intUnicodeMap"][29] = "29"
            self.computeModel["intUnicodeMap"][30] = "30"
            self.computeModel["intUnicodeMap"][31] = "31"
            self.computeModel["intUnicodeMap"][32] = "32"
            self.computeModel["intUnicodeMap"][33] = "33"
            self.computeModel["intUnicodeMap"][34] = "34"
            self.computeModel["intUnicodeMap"][35] = "35"
            self.computeModel["intUnicodeMap"][36] = "36"
            self.computeModel["intUnicodeMap"][37] = "37"
            self.computeModel["intUnicodeMap"][38] = "38"
            self.computeModel["intUnicodeMap"][39] = "39"
            self.computeModel["intUnicodeMap"][40] = "40"
            self.computeModel["intUnicodeMap"][41] = "41"
            self.computeModel["intUnicodeMap"][42] = "42"
            self.computeModel["intUnicodeMap"][43] = "43"
            self.computeModel["intUnicodeMap"][44] = "44"
            self.computeModel["intUnicodeMap"][45] = "45"
            self.computeModel["intUnicodeMap"][46] = "46"
            self.computeModel["intUnicodeMap"][47] = "47"
            self.computeModel["intUnicodeMap"][48] = "48"
            self.computeModel["intUnicodeMap"][49] = "49"
            self.computeModel["intUnicodeMap"][50] = "50"
            self.computeModel["intUnicodeMap"][51] = "51"
            self.computeModel["intUnicodeMap"][52] = "52"
            self.computeModel["intUnicodeMap"][53] = "53"
            self.computeModel["intUnicodeMap"][54] = "54"
            self.computeModel["intUnicodeMap"][55] = "55"
            self.computeModel["intUnicodeMap"][56] = "56"
            self.computeModel["intUnicodeMap"][57] = "57"
            self.computeModel["intUnicodeMap"][58] = "58"
            self.computeModel["intUnicodeMap"][59] = "59"
            self.computeModel["intUnicodeMap"][60] = "60"
            self.computeModel["intUnicodeMap"][61] = "61"
            self.computeModel["intUnicodeMap"][62] = "62"
            self.computeModel["intUnicodeMap"][63] = "63"
            self.computeModel["intUnicodeMap"][64] = "64"
            self.computeModel["intUnicodeMap"][65] = "65"
            self.computeModel["intUnicodeMap"][66] = "66"
            self.computeModel["intUnicodeMap"][67] = "67"
            self.computeModel["intUnicodeMap"][68] = "68"
            self.computeModel["intUnicodeMap"][69] = "69"
            self.computeModel["intUnicodeMap"][70] = "70"
            self.computeModel["intUnicodeMap"][71] = "71"
            self.computeModel["intUnicodeMap"][72] = "72"
            self.computeModel["intUnicodeMap"][73] = "73"
            self.computeModel["intUnicodeMap"][74] = "74"
            self.computeModel["intUnicodeMap"][75] = "75"
            self.computeModel["intUnicodeMap"][76] = "76"
            self.computeModel["intUnicodeMap"][77] = "77"
            self.computeModel["intUnicodeMap"][78] = "78"
            self.computeModel["intUnicodeMap"][79] = "79"
            self.computeModel["intUnicodeMap"][80] = "80"
            self.computeModel["intUnicodeMap"][81] = "81"
            self.computeModel["intUnicodeMap"][82] = "82"
            self.computeModel["intUnicodeMap"][83] = "83"
            self.computeModel["intUnicodeMap"][84] = "84"
            self.computeModel["intUnicodeMap"][85] = "85"
            self.computeModel["intUnicodeMap"][86] = "86"
            self.computeModel["intUnicodeMap"][87] = "87"
            self.computeModel["intUnicodeMap"][88] = "88"
            self.computeModel["intUnicodeMap"][89] = "89"
            self.computeModel["intUnicodeMap"][90] = "90"
            self.computeModel["intUnicodeMap"][91] = "91"
            self.computeModel["intUnicodeMap"][92] = "92"
            self.computeModel["intUnicodeMap"][93] = "93"
            self.computeModel["intUnicodeMap"][94] = "94"
            self.computeModel["intUnicodeMap"][95] = "95"
            self.computeModel["intUnicodeMap"][96] = "96"
            self.computeModel["intUnicodeMap"][97] = "97"
            self.computeModel["intUnicodeMap"][98] = "98"
            self.computeModel["intUnicodeMap"][99] = "99"
            self.computeModel["intUnicodeMap"][100] = "100"
            self.computeModel["intUnicodeMap"][101] = "101"
            self.computeModel["intUnicodeMap"][102] = "102"
            self.computeModel["intUnicodeMap"][103] = "103"
            self.computeModel["intUnicodeMap"][104] = "104"
            self.computeModel["intUnicodeMap"][105] = "105"
            self.computeModel["intUnicodeMap"][106] = "106"
            self.computeModel["intUnicodeMap"][107] = "107"
            self.computeModel["intUnicodeMap"][108] = "108"
            self.computeModel["intUnicodeMap"][109] = "109"
            print(self.computeModel)
    
            #define a jit compute instance with this model
            specs_model = {}
            specs_model['wordWeight'] =self.computeModel["wordWeight"]._dict_type
            specs_model['wordIndex'] = self.computeModel["wordIndex"]._dict_type
            specs_model['data'] = self.computeModel["data"]._dict_type
            specs_model['children'] = self.computeModel["children"]._dict_type
            specs_model['intUnicodeMap'] = self.computeModel["intUnicodeMap"]._dict_type
            specs_model['bowSize'] = types.int64
            @nb.jitclass(specs_model)
            class VisionModelEvaluator:
                def getEmptyIntList(self):
                    l = List()
                    l.append(12)
                    l.clear()
                    return l
                
                def getPhrase(self, vecStack):
                    numberOfVecs = vecStack.shape[0]
                    outBowPhrase = np.zeros(shape=(1, self.bowSize), dtype=types.float64)
                    for i in range(numberOfVecs):
                        _vec = np.zeros(shape=(vecStack.shape[1]), dtype=vecStack.dtype)
                        _vec = vecStack[i]
                        wordId = self.getWordId(_vec)
                        #get weight
                        wordWeight = 1
                        if wordId in self.wordWeight:
                            wordWeight = self.wordWeight[wordId]
                        #get vector
                        wordIndex = self.wordIndex[wordId]
                        word = np.zeros(shape=(1, self.bowSize), dtype=types.float64)
                        word[0][wordIndex] = wordWeight
                        outBowPhrase = outBowPhrase + word
                    sumBow = np.sum(outBowPhrase) # jk fixme todo
                    #print("sumBow", sumBow)
                    #sumBow = 1
                    bowVec = outBowPhrase / sumBow
                    return bowVec
                
                def getPhraseDistance(self, v1, v2):
                    return 1 - (0.5 * np.sum((v1/np.sum(v1))-(v2/np.sum(v2))))
                
                def getWordId(self, vec, above = None):
                    above = self.getEmptyIntList()
                    entryPoint = "root"
                    while (True):       
                        if entryPoint not in self.data:
                            return entryPoint
                        levelData = self.data[entryPoint]
                        _lenLevelData = levelData.shape[0]
                        _lenFeature = levelData.shape[1]
                        testColumn = np.zeros(shape=(_lenLevelData, _lenFeature), dtype=levelData.dtype)
                        for i in range(0, _lenLevelData):
                            testColumn[i] = vec
                        pairwiseDistance =  (testColumn != levelData).sum(axis=1)
                        bestChild = np.argmin(pairwiseDistance)
                        above.append(bestChild)
                        bestChildId = ""
                        lenAbove = len(above)
                        for i in range(lenAbove):
                            if (i + 1 == lenAbove):
                                bestChildId = bestChildId + self.intUnicodeMap[above[i]]
                            else:
                                bestChildId = bestChildId + self.intUnicodeMap[above[i]] + "-"
                        entryPoint = bestChildId 
            

    
                def __init__(self, wordWeight, wordIndex, data, _children, intUnicodeMap, bowSize):
                    self.wordWeight = wordWeight
                    self.wordIndex = wordIndex
                    self.data = data
                    self.children = _children
                    self.intUnicodeMap = intUnicodeMap
                    self.bowSize = bowSize
    
            # load the computeModel from the fileModel
            wordWeights = self.fileModel["wordWeights"]
            wordWeightKeys = wordWeights.keys()
            for wordWeightKey in wordWeightKeys:
                self.computeModel["wordWeight"][wordWeightKey] = wordWeights[wordWeightKey]

            wordIndex = self.fileModel["wordIndex"]
            wordIndexKeys = wordIndex.keys()
            for wordIndexKey in wordIndexKeys:
                self.computeModel["wordIndex"][wordIndexKey] = wordIndex[wordIndexKey]
                
            childrenMap = self.fileModel["children"]
            childrenMapKeys = childrenMap.keys()
            for childMapKey in childrenMapKeys:
                listOfChildren = List()
                for child in childrenMap[childMapKey]:
                    listOfChildren.append(child)
                self.computeModel["children"][childMapKey] = listOfChildren

            data = self.fileModel["data"]
            dataKeys = data.keys()
            for dataKey in dataKeys:
                point = np.asarray(data[dataKey])
                if point.shape[0] != 0:
                    self.computeModel["data"][dataKey] = point
            visionModelEvalInstance = VisionModelEvaluator(
                self.computeModel["wordWeight"], 
                self.computeModel["wordIndex"],
                self.computeModel["data"],
                self.computeModel["children"],
                self.computeModel["intUnicodeMap"],
                len(wordIndexKeys)
            )
            self.engine = visionModelEvalInstance
a = VisionModelPacker('visionModel.pickle')
cap = cv2.VideoCapture(0)
orb = cv2.ORB_create(edgeThreshold=2, patchSize=100, nlevels=15, fastThreshold=8, nfeatures=1000000, scoreType=cv2.ORB_FAST_SCORE, firstLevel=0)

def computeAndDisplayPhraseAndFrame(computeEngine, input_img):
    _kp1, _des1 = orb.detectAndCompute(input_img, None)
    _frameWithFeatures = cv2.drawKeypoints(input_img,_kp1,color=(0,255,0), outImage=None, flags=0)
    if _des1 is None:
        return
    phrase = computeEngine.getPhrase(_des1)
    lenFeatures = phrase.shape[1]
    _x = []
    _y = []
    for i in range(lenFeatures):
        _x.append(phrase[0][i])
        _y.append(i)
    #_fig, _axs = plt.subplots(2)
    #_axs[0].imshow(_frameWithFeatures[:,:,::-1])
    #_axs[1].plot(_x, _y)
    plt.figure(1)
    plt.clf()
    plt.plot(_x, _y)
    cv2.imshow("frame", _frameWithFeatures) #[:,:,::-1]
    return phrase

oldPhrase = None
iScore = 0
xScore = []
yScore = []
while (True):
    _, input_img = cap.read()
    phrase = computeAndDisplayPhraseAndFrame(a.engine, input_img)
    plt.pause(0.01)
    #print(np.count_nonzero(phrase))
    #print(phrase.shape)
    #print(phrase[phrase!=0].shape)
    if oldPhrase is not None:
        xScore.append(iScore)
        iScore = iScore + 1
        diff = 1 - (0.5 * np.sum(np.abs((phrase) - (oldPhrase)))) # 1 - (0.5 * np.abs()))
        yScore.append(diff)
        plt.figure(2)
        plt.clf()
        plt.plot(xScore, yScore)
        #print(np.count_nonzero(phrase - oldPhrase))
         #/np.sum(phrase) #/np.sum(oldPhrase)
        #print(diff)
        #print(np.count_nonzero(diff))
        #print(np.sum(diff))
        #print(diff.shape)
        #print(np.count_nonzero((phrase/np.sum(phrase))-(oldPhrase/np.sum(oldPhrase))))
        #print(1 - (0.5 * np.sum((phrase/np.sum(phrase))-(oldPhrase/np.sum(oldPhrase)))))
        #print(a.engine.getPhraseDistance(phrase, oldPhrase))
        #print("######")
    oldPhrase = phrase
    # Wait for 25ms
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #read_single_keypress()
print("done")
plt.show()
