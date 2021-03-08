from pyspark.sql import functions as F
import pyspark
import numpy as np
import sys

def seedClusters(data, k, metric): 
    centeroids = list(np.random.choice(data.shape[0], 1, replace=False)) 
    for _ in range(k - 1):
        distances = []
        for i in range(data.shape[0]): 
            point = data[i, :] 
            minDistance = sys.maxsize
            for j in range(len(centeroids)): 
                distance = metric(point, data[centeroids[j]]) 
                minDistance = min(minDistance, distance) 
            distances.append(minDistance) 
        distances = np.array(distances) 
        centeroids.append(np.argmax(distances)) 
        distances = [] 
    return centeroids

def nearestCenteroid(dataIdValue, centeroidIdValues, metric):
    import numpy as np
    dataId, dataValue = dataIdValue
    dataNp = np.asarray(dataValue)
    distances = []
    for centeroidIdValue in centeroidIdValues:
        centeroidId, centeroidValue = centeroidIdValue
        centeroidNp = np.asarray(centeroidValue)
        distance = metric(dataNp, centeroidNp)
        distances.append(distance)
        #distances.append((dataNp != centeroidNp).sum())
    distances = np.asarray(distances)
    closestCenteroid = np.argmin(distances)
    return int(closestCenteroid)

def optimiseClusterMembershipSpark(data, dataFrame, n, metric, intitalClusterIndices=None):
    dataShape = data.shape
    dataRDD = dataFrame.rdd
    lengthOfData = dataShape[0]
    if intitalClusterIndices is None:
        index = np.random.choice(lengthOfData, n, replace=False)
    else:
        index = intitalClusterIndices
    listIndex = [int(i) for i in list(index)]
    centeroidIdValues = [(i,data[index[i]]) for i in range(len(index))]
    dataRDD = dataRDD.filter(lambda dataIdValue: int(dataIdValue["id"]) not in listIndex)
    associatedClusterPoints = dataRDD.map(lambda dataIdValue: (dataIdValue[0],nearestCenteroid(dataIdValue, centeroidIdValues, metric)))
    clusters = associatedClusterPoints.toDF(["id", "bestC"]).groupBy("bestC").agg(F.collect_list("id").alias("cluster"))
    return index, clusters

def costKernel(data, testCenteroid, clusterData, metric):
    cluster = np.asarray(clusterData)
    lenCluster = cluster.shape[0]
    lenFeature = data.shape[1]
    testCenteroidColumn = np.zeros(shape=(lenCluster, lenFeature), dtype=data.dtype)
    newClusterColumn = np.zeros(shape=(lenCluster, lenFeature), dtype=data.dtype)
    for i in range(0, lenCluster):
        newClusterColumn[i] = data[cluster[i]]
        testCenteroidColumn[i] = data[int(testCenteroid)] 
    pairwiseDistance = metric(newClusterColumn, testCenteroidColumn)
    cost = np.sum(pairwiseDistance)
    return float(cost)

def optimiseCentroidSelectionSpark(data, dataFrame, centeroids, clustersFrames, metric):
    dataRDD = dataFrame.rdd
    dataShape = data.shape
    newCenteroidIds = []
    totalCost = 0
    for clusterIdx in range(len(centeroids)):
        print("clusterIdx", clusterIdx)
        oldCenteroid = centeroids[clusterIdx]
        clusterFrame = clustersFrames.filter(clustersFrames.bestC == clusterIdx).select(F.explode(clustersFrames.cluster))
        clusterData = clusterFrame.collect()[0]
        cluster = np.asarray(clusterData)
        costData = clusterFrame.rdd.map(lambda pointId: (pointId[0], costKernel(data, pointId[0], clusterData, metric)))
        cost = costData.map(lambda pointIdCost: pointIdCost[1]).sum()
        totalCost = totalCost + cost
        bestPoint = costData.sortBy(lambda pointId_Cost: pointId_Cost[1]).take(1)[0][0]
        newCenteroidIds.append(bestPoint)
    return (newCenteroidIds, totalCost)

def clusterOpt(data, dataFrame, nRegions, pointMetric, vectorMetric):
    # define a routine to keep going until cost stays the same or gets worse
    #get seeds
    seeds = seedClusters(data, nRegions, pointMetric)
    print(seeds)
    lastCenteroids, lastClusters = optimiseClusterMembershipSpark(data, dataFrame, nRegions, vectorMetric, seeds)
    lastCost = float('inf')
    iteration = 0
    escape = False
    while not escape:
        iteration = iteration + 1
        currentCenteroids, currentCost = optimiseCentroidSelectionSpark(lastCenteroids, lastClusters, data, dataFrame, vectorMetric)
        currentCenteroids, currentClusters = optimiseClusterMembershipSpark(data, dataFrame, nRegions, vectorMetric, currentCenteroids)
        print((currentCost<lastCost, currentCost, lastCost, currentCost - lastCost))
        if (currentCost<lastCost):
            print(("iteration",iteration,"cost improving...", currentCost, lastCost))
            lastCost = currentCost
            lastCenteroids = currentCenteroids
            lastClusters = currentClusters
        else:
            print(("iteration",iteration,"cost got worse or did not improve", currentCost, lastCost))
            escape = True
        print("--------------------")
    return (lastCenteroids, lastClusters)

#vector metrics
def hammingVector(stack1, stack2):
    return (stack1 != stack2).sum() #.sum(axis=1)
def euclideanVector(stack1, stack2):
    return (np.absolute(stack2-stack1)).sum() #.sum(axis=1)
# point metrics
def euclideanPoint(p1, p2): 
    return np.sum((p1 - p2)**2) 
def hammingPoint(p1, p2): 
    return np.sum((p1 != p2))

def fit(sc, data, nRegions = 2, metric = "euclidean", seeding = "heuristic"):
    if metric == "euclidean":
        pointMetric = euclideanPoint
        vectorMetric = euclideanVector
    elif metric == "hamming":
        pointMetric = hammingPoint
        vectorMetric = hammingVector
    else:
        print("unsuported metric")
        return

    dataN = np.asarray(data)
    seeds = None
    if (seeding == "heuristic"):
        seeds = seedClusters(dataN, nRegions, pointMetric)
    dataFrame  = sc.parallelize(data).zipWithIndex().map(lambda xy: (xy[1],xy[0])).toDF(["id", "vector"])
    lastCenteroids, lastClusters = optimiseClusterMembershipSpark(dataN, dataFrame, nRegions, vectorMetric, seeds)
    lastCost = float('inf')
    iteration = 0
    escape = False
    while not escape:
        iteration = iteration + 1
        currentCenteroids, currentCost = optimiseCentroidSelectionSpark(dataN, dataFrame, lastCenteroids, lastClusters, vectorMetric)
        currentCenteroids, currentClusters = optimiseClusterMembershipSpark(dataN, dataFrame, nRegions, vectorMetric, currentCenteroids)
        print((currentCost<lastCost, currentCost, lastCost, currentCost - lastCost))
        if (currentCost<lastCost):
            print(("iteration",iteration,"cost improving...", currentCost, lastCost))
            lastCost = currentCost
            lastCenteroids = currentCenteroids
            lastClusters = currentClusters
        else:
            print(("iteration",iteration,"cost got worse or did not improve", currentCost, lastCost))
            escape = True
    return (lastCenteroids, lastClusters)