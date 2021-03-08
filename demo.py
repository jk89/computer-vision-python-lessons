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
import math
from numba import njit, prange
from datetime import datetime, timedelta
# program needed 
maxK = 110
epoch = datetime.utcfromtimestamp(0)
temporalPoolingInterval = int((1 * 10^9) * (5)) #5 seconds

# program behaviour variables
skipExpensiveGraphing = False# False

# kp des examples and types
desExample = np.asarray([[0.1, 0.1],[0.1, 0.1]], dtype=np.uint8)
kpsExample = np.asarray([[210,  13,  80, 128, 155, 224,  72, 120, 234, 104,  92,  75, 232, 141, 216, 238,  34, 172,
 198, 218, 191, 113, 205, 136,  88, 150, 203,  57, 204,  91, 129,  83],[210,  13,  80, 128, 155, 224,  72, 120, 234, 104,  92,  75, 232, 141, 216, 238,  34, 172,
 198, 218, 191, 113, 205, 136,  88, 150, 203,  57, 204,  91, 129,  83]], dtype=np.float32)
typeOfKpsExample = typeof(kpsExample)
typeOfDesExample = typeof(desExample)

#define useful metrics and minimisation functions

@njit
def hammingVector(stack1, stack2):
    return (stack1 != stack2).sum(axis=1)

#load data into JIT class
@njit
def compareLevelData(levelData, vec, metric=hammingVector):
    lenLevelData = levelData.shape[0]
    lenFeature = levelData.shape[1]
    testColumn = np.zeros(shape=(lenLevelData, lenFeature), dtype=levelData.dtype)
    for i in range(0, lenLevelData):
        testColumn[i] = vec
        pairwiseDistance =  metric(levelData, vec)
        minIndex = np.argmin(pairwiseDistance)
    return np.argmin(pairwiseDistance)

#define utils


@nb.jit(nopython=True)
def getEmptyStringList():
    y = List()
    y.append("ssss") 
    y.clear()
    return y

@nb.jit(nopython=True)
def getEmptyIntList():
    l = List()
    l.append(12)
    l.clear()
    return l

@nb.jit(nopython=True)
def getEmptyFloat64List():
    l = List()
    l.append(12.1212)
    l.clear()
    return l

@nb.jit(nopython=True)
def getCoordinatesThing():
    il = List()
    il.append(23.3)
    il.append(23.2)
    li = List()
    li.append(il)
    li.append(il)
    l = List()
    l.append(li)
    l.clear()
    return l

@nb.jit(nopython=True)
def getIndexPairsThing():
    il = List()
    il.append(23)
    il.append(21)
    l = List()
    l.append(il)
    l.clear()
    return l

####DATABASE TYPES#####

#define id type
stringListExample = List()
stringListExample.append("s")
stringListExample.clear()
stringListExampleType = typeof(stringListExample)

# define string: float64 map
emptyStringFloat64Map = typed.Dict.empty(
                key_type = types.string,
                value_type = types.float64
)

emptyStringStringMap = typed.Dict.empty(
                key_type = types.string,
                value_type = types.string
)

emptyStringFloat64MapType = emptyStringFloat64Map._dict_type

emptyInt64StringMap = typed.Dict.empty(
                key_type = types.int64,
                value_type = types.string
)

emptyInt64Float64Map = typed.Dict.empty(
                key_type = types.int64,
                value_type = types.float64
)
bowExample = np.asarray([0.1, 0.1, 0.2], dtype = np.float64)
bowExampleType = typeof(bowExample)

neighbouringWordsExample = typed.Dict.empty(
                                    key_type = types.string,
                                    value_type = types.string #stringListExampleType
)

### FRAME DEF ###
#define the frame details class
spec = [
    ('kps', typeof(kpsExample)),   
    ('des', typeof(desExample)),
    ('bow', bowExampleType),
    ('neighbouringWords', neighbouringWordsExample._dict_type)
]
@jitclass(spec)
class Frame(object):
    def __init__(self, kps, des, bow, neighbouringWords):
        #self.frameLocation = frameLocation
        self.kps = kps
        self.des = des
        self.bow = bow
        self.neighbouringWords = neighbouringWords
                    #frame [frameId]
                    ###
                    # [wordId1, wordId2]
                    # [wordId] : [siblingIds...] depending on l if l=0 siblings l=1 cousins etc
                    # bowVec


### WORD DEF ###

"""
                    #word [wordId]
                    # [frameId] : [weightInImage]
                    # [frameId1, frameId2]
                    """
#define the word class
intListExampleType = typeof(getEmptyIntList())
int64Float64MapType = emptyInt64Float64Map._dict_type
spec = [
    ('frameIds', intListExampleType),
    ('frameWeights', int64Float64MapType)
]
@jitclass(spec)
class WordData(object):
    def __init__(self, frameIds, frameWeights):
        self.frameIds = frameIds
        self.frameWeights = frameWeights
        pass

def getEmptyWordDataInst():
    emptyWeights = typed.Dict.empty(key_type = types.int64, value_type = types.float64)
    emptyIds = getEmptyIntList()
    return WordData(emptyIds, emptyWeights)



#load the model
model = None
with open('models/visionModel.pickle', 'rb') as handle:
    model = pickle.load(handle)



#define the lass which packs the compute model and has higher level functions

class VisionModelPacker:
    def __init__(self, filename=None):
        #define cv2 utils
        self.cap = cv2.VideoCapture(0) # 1500 2000 #300
        self.orb = cv2.ORB_create(edgeThreshold=10, patchSize=100, nlevels=15, fastThreshold=10, nfeatures=300, scoreType=cv2.ORB_FAST_SCORE, firstLevel=0)
        self.old_kp1 = None
        self.old_img = None
        self.old_kp1_np = None
        self.R = None
        self.t = None
        #load model... or dont
        if filename is not None:
            self.loadModel(filename)
            self.packVisionModelEvaluator()
        else:
            self.fileModel = None

    def initComputeModel(self):
        print("Initialising compute model")
        #define compute model
        self.computeModel = {
            "wordData": typed.Dict.empty(
                key_type = types.string,
                value_type = WordData.class_type.instance_type
            ),
            "wordWeight": typed.Dict.empty(
                key_type=types.string,
                value_type=types.float64,
            ),
            "wordIndex": typed.Dict.empty(
                key_type=types.string,
                value_type=types.int64
            ),
            "wordInverseIndex": typed.Dict.empty(
                key_type=types.int64,
                value_type=types.string,
            ),
            "data": typed.Dict.empty(
                        key_type=types.string,
                        value_type=typeOfKpsExample,
            ),
            "children": typed.Dict.empty(
                        key_type=types.string,
                        value_type=nb.types.ListType(types.string),
            ),
            "intUnicodeMap": typed.Dict.empty(
                        key_type=types.int64,
                        value_type=types.string,
            )
        }
        for i in range(maxK):
            self.computeModel["intUnicodeMap"][i] = str(i)
  
    def initDatabaseModel(self):
        #init database
        self.database = {}
        self.database["wordData"] = typed.Dict.empty(
                key_type = types.string,
                value_type = WordData.class_type.instance_type
        )
        self.database["frameData"] = typed.Dict.empty(
                key_type = types.int64,
                value_type = Frame.class_type.instance_type
        )
        for wordId in self.wordWeightKeys:
            emptyWord = getEmptyWordDataInst()
            self.database["wordData"][wordId] = emptyWord

    def packModel(self):
        print("Packing computational model from file model")
        # load the computeModel from the fileModel
        wordWeights = self.fileModel["wordWeights"]
        wordWeightKeys = wordWeights.keys()
        self.wordWeightKeys = wordWeightKeys
        for wordWeightKey in wordWeightKeys:
            self.computeModel["wordWeight"][wordWeightKey] = wordWeights[wordWeightKey]

        wordIndex = self.fileModel["wordIndex"]
        self.wordIndexKeys = wordIndexKeys = wordIndex.keys()
        for wordIndexKey in wordIndexKeys:
            self.computeModel["wordInverseIndex"][wordIndex[wordIndexKey]] = wordIndexKey
            self.computeModel["wordIndex"][wordIndexKey] = wordIndex[wordIndexKey]    
            #print("wordIndwordInverseIndexex", self.computeModel["wordInverseIndex"])
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
            point = np.asarray(data[dataKey], dtype=kpsExample.dtype)
            if point.shape[0] != 0:
                self.computeModel["data"][dataKey] = point


    #open file and unpack
    def loadModel(self, filename):
        print("loading file model")
        with open('models/visionModel.pickle', 'rb') as handle:
            self.fileModel = pickle.load(handle)

    def timestampInt64(self):
        return int((datetime.now() - epoch).total_seconds() * 1000000)

    def processFrame(self):
        frameReadTime = self.timestampInt64()
        _, input_img = self.cap.read()
        _kp1, _des1 = self.orb.detectAndCompute(input_img, None)
        if type(_des1) is tuple or _kp1 is None or _des1 is None:
            return
        kp1 = cv2.KeyPoint_convert(_kp1)
        start = datetime.now() 
        ret = self.engine.processFrame(kp1, _des1, frameReadTime)
        if ret[0] is True: # we did an insert so save it to the frames folder
            cv2.imwrite("frames/" + str(frameReadTime) + ".jpg", input_img)
        else:
            if self.old_kp1 is not None:
                candidates = ret[1]
                src_pts = np.float32([ _kp1[m[0]].pt for m in candidates ]).reshape(-1,1,2)
                dst_pts = np.float32([ self.old_kp1[m[1]].pt for m in candidates ]).reshape(-1,1,2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                matchesMask = mask.ravel().tolist()
                filtered_src_pts = np.asarray(list(filter(lambda x: x is not None, [kp1[candidates[i][0]] if matchesMask[i] == 1 else None for i in range(len(matchesMask))])))
                filtered_dst_pts = np.asarray(list(filter(lambda x: x is not None, [self.old_kp1_np[candidates[i][1]] if matchesMask[i] == 1 else None for i in range(len(matchesMask))])))
                E, mask = cv2.findEssentialMat(filtered_src_pts, filtered_dst_pts, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=3.0)
                points, R, t, mask = cv2.recoverPose(E, filtered_src_pts, filtered_dst_pts)

                if self.R is None:
                    self.R = R
                    self.t = t
                elif self.R.shape == R.shape and self.t.shape == t.shape:
                    self.R = self.R + R
                    self.R[0][0] = 1
                    self.R[1][1] = 1
                    self.R[2][2] = 1
                    self.t = self.t + t
                print("tracked points", filtered_dst_pts.shape)
                print("mask essential matric", len(mask))
                print("self.R", self.R) #points mask
                print("self.t", self.t) #points mask
                #M_r = np.hstack((R, t))
                #M_l = np.hstack((np.eye(3, 3), np.zeros((3, 1))))

                #P_l = np.dot(K_l,  M_l)
                #P_r = np.dot(K_r,  M_r)
                #point_4d_hom = cv2.triangulatePoints(P_l, P_r, np.expand_dims(pts_l, axis=1), np.expand_dims(pts_r, axis=1))
                #point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
                #point_3d = point_4d[:3, :].T

        delta = datetime.now() - start
        self.old_kp1 = _kp1
        self.old_img = input_img
        self.old_kp1_np = kp1
        print("addImageToDB took " + str(delta.total_seconds()) + "(s)")
        return ret

    def computeAndDisplayPhraseAndFrame(self, input_img):
        if input_img is None:
            return (None, None)
        _kp1, _des1 = self.orb.detectAndCompute(input_img, None)
        _frameWithFeatures = cv2.drawKeypoints(input_img,_kp1,color=(0,255,0), outImage=None, flags=0)
        if _des1 is None:
            return (None, None)
        phrase, _ = self.engine.getPhrase(_des1)
        lenFeatures = phrase.shape[1]
        if skipExpensiveGraphing == False:
            _x = []
            _y = []
            for i in range(lenFeatures):
                _x.append(phrase[0][i])
                _y.append(i)
            plt.figure("Visual Phrase Vector")
            plt.clf()
            if (len(_x) == len(_y)):
                plt.plot(_x, _y)
        return (phrase, _frameWithFeatures)


    


    def packVisionModelEvaluator(self, featureExample = [[1],[1]]):
        if self.fileModel is None:
            print("Please load a model first")
        else:
            #define storage types
            self.initComputeModel()
            self.packModel()
            self.initDatabaseModel()

    
            print("Defining jit compute class")
            #define a jit compute instance with this model
            specs_model = {}
            specs_model['wordWeight'] =self.computeModel["wordWeight"]._dict_type
            specs_model['wordData'] = self.database["wordData"]._dict_type
            specs_model['wordIndex'] = self.computeModel["wordIndex"]._dict_type
            specs_model['wordInverseIndex'] = self.computeModel["wordInverseIndex"]._dict_type
            specs_model['data'] = self.computeModel["data"]._dict_type
            specs_model['children'] = self.computeModel["children"]._dict_type
            specs_model['intUnicodeMap'] = self.computeModel["intUnicodeMap"]._dict_type
            specs_model['bowSize'] = types.int64
            specs_model['frameData'] = self.database["frameData"]._dict_type
            specs_model['lastKeyFrameInsertion'] = types.int64
            specs_model['count'] = types.int64
            specs_model['lastPhrase'] = bowExampleType
            specs_model['lastWords'] = stringListExampleType
            specs_model['lastKps'] = typeOfKpsExample 
            @nb.jitclass(specs_model)
            class VisionModelEvaluator:


                def findCommonWordIds(self, ancestorNodeId):
                    #ancestorNodeId like "0-1"
                    cont = True
                    nodesToFollow = List()
                    nodesToFollow.append(ancestorNodeId)
                    words = getEmptyStringList()
                    while (cont):
                        nodeToFollow = nodesToFollow.pop()
                        if nodeToFollow in self.children:
                            descendants = self.children[nodeToFollow]
                            for descendant in descendants:
                                if descendant in self.children:
                                    nodesToFollow.append(descendant)
                                else:
                                    if descendant in self.data:
                                        lastLevelData = self.data[descendant]
                                        lenData = lastLevelData.shape[0]
                                        for i in range(lenData):
                                            wordId = descendant+"-"+self.intUnicodeMap[i]
                                            self.wordIndex[wordId]
                                            words.append(wordId)
                                    else:
                                        self.wordIndex[descendant]
                                        words.append(descendant)
                        else:
                            if nodeToFollow in self.data:
                                thisWordsData = self.data[nodeToFollow]
                                lenData = thisWordsData.shape[0]
                                for i in range(lenData):
                                    wordId = nodeToFollow+"-"+self.intUnicodeMap[i]
                                    self.wordIndex[wordId]
                                    words.append(wordId)
                            else:
                                if nodeToFollow in self.wordIndex:
                                    words.append(nodeToFollow)
                                else:
                                    raise ValueError("asd")
                                #self.wordIndex[nodeToFollow]
                                
                        if len(nodesToFollow) == 0:
                            cont = False
                    return words

                def searchDB(self, kps, des, currentPhrase, frameReadTime, wordList, lastPhrase):
                    normalisedScoreThreshold = 0.3
                    #print("currentPhrase", currentPhrase)
                    #print("lastPhrase", self.lastPhrase)
                    approximateExpectedScore = self.getPhraseDistance(currentPhrase, lastPhrase)
                    if approximateExpectedScore < 0.1: #if the is little correspondance to the last frame skip search #camera has too much motion
                        return
                    #scan frames bow self.frameData[self.count]
                    keyFrameKeys = self.frameData.keys()


                    #define an island
                    #island score
                    islands = typed.Dict.empty(
                                    key_type = types.int64,
                                    value_type = intListExampleType # list of int64s
                    )

                    scores = typed.Dict.empty(
                                    key_type = types.int64,
                                    value_type = types.float64
                    )

                    islandScore = typed.Dict.empty(
                                    key_type = types.int64,
                                    value_type = types.float64
                    )
                    
                    #temporal binning and framescore computation
                    keyFrameIslandId = 0
                    for keyFrameId in keyFrameKeys:
                        keyFrameBOW = self.frameData[keyFrameId].bow
                        frameBOWDistance = self.getPhraseDistance(keyFrameBOW, currentPhrase)
                        normalisedScore = frameBOWDistance / approximateExpectedScore
                        if (normalisedScore > normalisedScoreThreshold) and ((keyFrameIslandId > (keyFrameIslandId + temporalPoolingInterval)) or keyFrameIslandId == 0):
                            keyFrameIslandId = keyFrameId
                            islands[keyFrameIslandId] = getEmptyIntList()
                            islands[keyFrameIslandId].append(keyFrameIslandId)
                            scores[keyFrameIslandId] = normalisedScore
                            islandScore[keyFrameIslandId] = normalisedScore
                        elif normalisedScore > normalisedScoreThreshold:
                            islands[keyFrameIslandId].append(keyFrameId)
                            scores[keyFrameId] = normalisedScore
                            islandScore[keyFrameIslandId] = islandScore[keyFrameIslandId] + normalisedScore

                    bestIslandScore = 0
                    bestIslandId = 0
                    for islandId in islandScore.keys():
                        if islandScore[islandId] > bestIslandScore:
                            bestIslandId = islandId
                            bestIslandScore = islandScore[islandId]
                    print("best island", bestIslandScore, bestIslandId)

                    #check consistancy with k- previous queries??
                    #self.lastQueryIsland
                    # check last k intervals before now exist and have frames in them #frameReadTime

                    #best frame
                    bestFrameId = 0
                    bestFrameScore = 0
                    for frameId in islands[bestIslandId]:
                        if scores[frameId] > bestFrameScore:
                            bestFrameId = frameId
                            bestFrameScore = scores[frameId]

                    #geometric check
                    #neighbouringWords
                    bestMatchFrameData = self.frameData[frameId]
                    # wordList
                    matches = 0
                    total = len(wordList)
                    for wordId in wordList:
                        #check for correspondances
                        if wordId in bestMatchFrameData.neighbouringWords:
                            matches = matches + 1
                    if (matches != total):
                        return
                    print("matches", matches, "total", total)
                    pass
                    # we have a loop closure
                    #perform Bundle adjustment


                    

                def processFrame(self, kps, des, frameReadTime):

                    lastPhrase = self.lastPhrase
                    lastWords = self.lastWords
                    lastKps = self.lastKps
                    _currentPhrase, currentWordList = self.getPhrase(des)
                    if _currentPhrase is not None and kps is not None:
                        currentPhrase = _currentPhrase[0]
                    else:
                        return (False, getIndexPairsThing())

                    self.lastPhrase = currentPhrase
                    self.lastWords = currentWordList
                    self.lastKps = kps

                    #if 20 frames since the last insert
                    #More than 20 frames since last global relocalization ??
                    #Current frame tracks less than 90% than that of the reference keyframe ??
                    #Local mapping is idle, another thread
                    countVsLastInsertCount = (self.count - self.lastKeyFrameInsertion)
                    triggers = countVsLastInsertCount > 20

                    #Current frame tracks at least 50 points
                    #AND kps.shape[0] > 50: 
                    #    pass #test this
                    # has atleast one previous frame
                    required = kps.shape[0] > 50 #self.count > 1 and 

                    #start collecting lastPhrase for normalisation
                    #if not triggers and not required: #countVsLastInsertCount > 19 and
                    #    #print("setting lastPhrase")
                    #    lastPhrase, _ = self.getPhrase(des)
                    #    if lastPhrase is not None:
                    #        self.lastPhrase = lastPhrase[0]

                    
                    



                    self.count = self.count + 1

                    #if insert keyframe criteria are satisified
                    #print("triggrs, required",triggers, required)
                    if (triggers and required):
                        self.addImageToDB(currentPhrase, kps, des, frameReadTime)
                        if self.lastKeyFrameInsertion != 0: #only if we have a keyFrame
                            self.searchDB(kps, des, currentPhrase, frameReadTime, currentWordList, lastPhrase)
                            #global relocalations
                        return (True, getIndexPairsThing())
                    else:
                        return (False, self.findMatches(kps, lastKps, currentWordList, lastWords))

                def findMatches(self, cKps, oldKps, cWords, oldWords):
                    lencWords = len(cWords)
                    candidatesList = getIndexPairsThing()
                    for cWordIndex in range(lencWords):
                        cWord = cWords[cWordIndex]
                        nearbyWords = self.findWordNeighbourhood(cWord)
                        #any of these words existing in oldWords are a candidate match pair
                        for nearbyWord in nearbyWords:
                            #scan oldWords for matches
                            if nearbyWord in oldWords:
                                oldWordIndex = oldWords.index(nearbyWord)
                                pair = getEmptyIntList()
                                pair.append(cWordIndex)
                                pair.append(oldWordIndex)
                                #new = getEmptyFloat64List()
                                #new.append(cKps[cWordIndex][0])
                                #new.append(cKps[cWordIndex][1])
                                #old = getEmptyFloat64List()
                                #old.append(oldKps[oldWordIndex][0])
                                #old.append(oldKps[oldWordIndex][1])
                                #pair = List()
                                #pair.append(new)
                                #pair.append(old)
                                #pair.append()
                                #pair.append(oldKps[oldWordIndex])
                                candidatesList.append(pair)
                            pass
                        pass
                    pass
                    return candidatesList

                def findWordNeighbourhood(self, wordId, nearestAncestorDistance=1):
                    ancestorNodePath = wordId.split("-")[:-nearestAncestorDistance]
                    if len(ancestorNodePath) == 0:
                        ancestorNodePath = ["root"]
                        #root case?
                        pass
                    ancestorNodeId = "-".join(ancestorNodePath)
                    return self.findCommonWordIds(ancestorNodeId)
                    


                def addImageToDB(self, currentPhrase, kps, des, frameReadTime):
                    print("DB INSERT")
                    phrase = currentPhrase
                    if phrase is None:
                        return
                    #print(self.wordInverseIndex)

                    """
                    correspondences only between those features that belong
                    to the same words, or to words with common ancestors at level l.
                    """
                    #print(self.children)
                    #print(self.wordIndex)
                    #print(self.data)
                    neighbouringWords = typed.Dict.empty(
                                    key_type = types.string,
                                    value_type = types.string
                    )
                    nearestAncestorDistance = 1 #getEmptyStringList()
                    for i in range(self.bowSize):
                        # phrase[i] is the weight
                        wordId = self.wordInverseIndex[i]
                        wordWeightInImage = phrase[i]
                        self.wordData[wordId].frameIds.append(self.count)
                        self.wordData[wordId].frameWeights[self.count] = wordWeightInImage # might not need this jk remove
                        #self.
                        #self.frameIds = frameIds
        #self.frameWeights = frameWeights

                        ancestorNodePath = wordId.split("-")[:-nearestAncestorDistance]
                        if len(ancestorNodePath) == 0:
                            ancestorNodePath = ["root"]
                            #root case?
                            pass
                        ancestorNodeId = "-".join(ancestorNodePath)
                        commonWords = self.findCommonWordIds(ancestorNodeId)
                        for neighbour in commonWords:
                            neighbouringWords[neighbour] = wordId
                            #if neighbour not in neighbouringWords:
                            #    l = getEmptyStringList()
                            #    l.append(wordId)
                            #    neighbouringWords[neighbour] = l
                            #else:
                            #    neighbouringWords[neighbour].append(wordId)
                        #print(commonWords)
                        #print(wordId, i)
                        # i is the wordIndex
                        pass
                    #print(neighbouringWords)
                    self.lastKeyFrameInsertion = self.count
                    self.frameData[frameReadTime] = Frame(kps, des, phrase, neighbouringWords)

                
                def getPhrase(self, vecStack):
                    numberOfVecs = vecStack.shape[0]
                    outBowPhrase = np.zeros(shape=(1, self.bowSize), dtype=types.float64)
                    wordIds = getEmptyStringList()
                    for i in range(numberOfVecs):
                        _vec = np.zeros(shape=(vecStack.shape[1]), dtype=vecStack.dtype)
                        _vec = vecStack[i]
                        wordId = self.getWordId(_vec)
                        wordIds.append(wordId)
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
                    return bowVec, wordIds
                
                def getPhraseDistance(self, v1, v2):
                    return 1 - (0.5 * np.sum(np.abs((v1) - (v2))))
                    #return 1 - (0.5 * np.sum((v1/np.sum(v1))-(v2/np.sum(v2))))
                
                def getWordId(self, vec, above = None):
                    above = getEmptyIntList()
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
            

    
                def __init__(self, wordWeight, wordIndex, wordInverseIndex, data, _children, intUnicodeMap, frameData, wordData, bowSize, lastPhraseExample, stringListExample, lastKps):
                    self.wordWeight = wordWeight
                    self.wordIndex = wordIndex
                    self.wordInverseIndex = wordInverseIndex
                    self.data = data
                    self.children = _children
                    self.intUnicodeMap = intUnicodeMap
                    self.bowSize = bowSize
                    self.frameData = frameData
                    self.count = 0
                    self.wordData = wordData
                    self.lastKeyFrameInsertion = 0
                    self.lastPhrase = lastPhraseExample
                    self.lastWords = stringListExample
                    self.lastKps = lastKps
    

            print("Initalising compute engine")
            visionModelEvalInstance = VisionModelEvaluator(
                self.computeModel["wordWeight"], 
                self.computeModel["wordIndex"],
                self.computeModel["wordInverseIndex"],
                self.computeModel["data"],
                self.computeModel["children"],
                self.computeModel["intUnicodeMap"],
                self.database["frameData"],
                self.database["wordData"],
                len(self.wordIndexKeys),
                bowExample,
                stringListExample,
                kpsExample
            )
            self.engine = visionModelEvalInstance

startLoad = datetime.now() 

#create this instance
a = VisionModelPacker('models/visionModel.pickle') #visionModelK4N50k-2.pickle

delta = datetime.now() - startLoad
print("Init took " + str(delta.total_seconds()) + "(s)")


#print(orb)
#print(input_img)
#_, input_img = cap.read()
#_kp1, _des1 = orb.detectAndCompute(input_img, None)
#kp1 = cv2.KeyPoint_convert(_kp1)
#print(kp1.shape, kp1.dtype)
#print(_des1.shape, _des1.dtype)
#print(typeof(kp1))
#print(typeof(_des1))
#p = Frame(kp1, _des1)
#print(p)
#a.engine.frames[1]
#while True:
#    z = a.processFrame()
#print(typeof(z))

def demo():
    print("running demo")
    oldPhrase = None
    iScore = 0
    xScore = []
    yScore = []
    records = []
    xRecords = []
    yRecords = []
    while (True):
        ini_time_for_now = datetime.now() 
        _, input_img = a.cap.read()
        (phrase, frameWithFeatures) = a.computeAndDisplayPhraseAndFrame(input_img)
        if (phrase is None):
            continue
        if skipExpensiveGraphing == False:
            plt.pause(0.01) #0.05
        if oldPhrase is not None:
            font                   = cv2.FONT_HERSHEY_SIMPLEX
            topRight = (270,25)
            fontScale              = 1
            fontColor              = (255,0,0)
            lineType               = 2
            cv2.putText(
                input_img, 
                str(datetime.now()), 
                topRight, 
                font, 
                fontScale,
                fontColor,
                lineType
            )
            records.append({"phrase": phrase, "img": input_img})
            xScore.append(iScore)
            iScore = iScore + 1
            if phrase is None:
                continue
            #diff = 1 - (0.5 * np.sum(np.abs((phrase) - (oldPhrase)))) # 1 - (0.5 * np.abs()))
            bestScore = 0
            bestIndex = 0
            xRecords = []
            yRecords = []
            for i in range(len(records[:-100])):
                xRecords.append(i)
                record = records[i]
                score = 1 - (0.5 * np.sum(np.abs((phrase) - (record["phrase"]))))
                if skipExpensiveGraphing == False:
                    yRecords.append(score)
                #print("score i", score, i, score < bestScore, bestIndex)
                if score > bestScore:
                    bestScore = score
                    bestIndex = i
            if skipExpensiveGraphing == False:
                yScore.append(bestScore)
                plt.figure("Best Match Score In Image Database")
                plt.clf()
                if len(xScore) == len(xScore):
                    #print(len(xScore), len(yScore))
                    plt.plot(xScore, yScore)
            imgCopy = records[bestIndex]["img"].copy()
            font                   = cv2.FONT_HERSHEY_SIMPLEX
            topLeft = (10,40)
            fontScale              = 1
            fontColor              = (0,255,0)
            lineType               = 3
            cv2.putText(
                imgCopy, 
                str(math.floor(bestScore * 100) / 100), 
                topLeft, 
                font, 
                fontScale,
                fontColor,
                lineType
            )
            cv2.imshow("bestMatch", imgCopy)
            #plt.figure(3)
            #plt.clf()
            #plt.plot(xRecords, yRecords)

        oldPhrase = phrase
        # Wait for 25ms
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        delta = datetime.now() - ini_time_for_now
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        topLeft = (10,40)
        fontScale              = 1
        fontColor              = (0,0,255)
        lineType               = 3
        cv2.putText(
            frameWithFeatures,
            str(math.floor(1 / (delta.total_seconds() ))), 
            topLeft, 
            font, 
            fontScale,
            fontColor,
            lineType
        )
        cv2.imshow("frame", frameWithFeatures)
        #read_single_keypress()
    print("done")
    plt.show()
"""
"""
"""
@njit(parallel=True)
def getPhrase(engine, vecStack):
    numberOfVecs = vecStack.shape[0]
    outBowPhrase = np.zeros(shape=(1, engine.bowSize), dtype=types.float64)
    wordList = np.zeros(shape=(numberOfVecs, engine.bowSize), dtype=types.float64)
    for i in prange(numberOfVecs):
        _vec = np.zeros(shape=(vecStack.shape[1]), dtype=vecStack.dtype)
        _vec = vecStack[i]
        wordId = engine.getWordId(_vec)
        ##get weight
        wordWeight = 1
        if wordId in engine.wordWeight:
            wordWeight = engine.wordWeight[wordId]
        #get vector
        wordIndex = engine.wordIndex[wordId]
        word = np.zeros(shape=(1, engine.bowSize), dtype=types.float64)
        word[0][wordIndex] = wordWeight
        wordList[i] = word
    for i in range(numberOfVecs):
        outBowPhrase = outBowPhrase + wordList[i]
    sumBow = np.sum(outBowPhrase) # jk fixme todo
    bowVec = outBowPhrase / sumBow
    return bowVec
"""


# useful interupt function
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
demo()