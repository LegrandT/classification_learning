from sklearn import datasets
from math import *
import operator
import random
import matplotlib.pyplot as plt

iris = datasets.load_iris()


# def binDi(x, n, p):
#     return (factorial(n) / (factorial(n - x) * factorial(x))) * pow(p, x) * pow(1 - p, n - x)


def getLabel(train):
    labels = []
    for i in range(len(train)):
        if train[i] not in labels:
            labels.append(train[i])
    return labels

def eucliDist(a, b):
    d = 0
    for i in range(len(a)):
        d += pow((a[i] - b[i]), 2)
    return sqrt(d)

def kmean(xtrain, k):

    # k is the number of classes

    # create k centers randomly, in the range of max and min of each attributes
    bounds = list(map(list, zip(*xtrain)))
    centers = []
    for i in range(k):
        centers.append([])
        for j in range(len(xtrain[0])):
            centers[i].append(random.uniform(min(bounds[j]), max(bounds[j])))

    while True:
        distances = []

        # keep old centers in memory to later calculate difference with new centers
        oldCenters = []
        for i in range(len(centers)):
            oldCenters.append([])
            for j in range(len(centers[0])):
                oldCenters[i].append(centers[i][j])
        clusters = []

        # for each point, calculate the distance with each center
        for i in range(len(xtrain)):
            temp = []
            for j in range(len(centers)):
                temp.append(eucliDist(xtrain[i], centers[j]))
            distances.append(temp)

        # temporary label all training data with there closest center
        for i in range(len(centers)):
            clusters.append([])
        for i in range(len(xtrain)):
            for j in range(len(centers)):
                if distances[i][j] == min(distances[i]):
                    clusters[j].append(xtrain[i])

        # move the class center in the mean of the points in it's class
        for i in range(len(clusters)):
            clusters[i] = list(map(list, zip(*clusters[i])))
        for i in range(len(centers)):
            for j in range(len(centers[0])):
                if len(clusters[i]) == 0:
                    for l in range(len(centers[0])):
                        centers[i][l] = random.uniform(min(bounds[l]), max(bounds[l]))
                    break
                else:
                    centers[i][j] = sum(clusters[i][j])/len(clusters[i][j])

        # if difference between old and new centers is lower that threshold, center are considerate found
        change = 0
        for i in range(len(centers)):
            change += eucliDist(oldCenters[i], centers[i])
        if change < 0.00000001:
            return centers


def kmeancl(centers, x):
    euclids = []
    for i in range(len(centers)):
        euclids.append(eucliDist(centers[i], x))
    return euclids.index(min(euclids))


# to check what .. is the right one
def kmeanMaxAcc(predict, acc):
    accuracies = []
    for l in range(len(predict[0])-1):
        for j in range(len(predict)):
            temp = predict[j][0]
            predict[j][0] = predict[j][1]
            predict[j][1] = temp
        for j in range(len(predict[0])):
            for i in range(len(predict)):
                temp = predict[i].pop(0)
                if temp == 1:
                    predict[i].append(1)
                else:
                    predict[i].append(0)
            accuracies.append(Accuracy(acc, predict))
    return max(accuracies)


def muCl(xtraining, ytraining, x, knear):
    nearest = {}
    for i in range(len(xtraining)):
        nearest[i] = eucliDist(xtraining[i], x)
    nearest = sorted(nearest.items(), key=operator.itemgetter(1))

    classes = getLabel(ytraining)
    classcount = [0]*len(classes)
    for i in range(len(ytraining)):
        for j in range(len(classes)):
            if ytraining[i] == classes[j]:
                classcount[j] +=1

    kcount = [0]*len(classes)
    for i in range(knear):
        for j in range(len(classes)):
            # print(i, knear, nearest[i][0], ytraining[nearest[i][0]], classes[j])
            if ytraining[nearest[i][0]] == classes[j]:
                kcount[j] += 1

    kprob = [0]*len(classes)
    for i in range(len(kprob)):
        kprob[i] = ((kcount[i]/classcount[i])*(classcount[i]/len(ytraining))) # /knear/len(xtraining)

    result = classes[kprob.index(max(kprob))]

    # result = []
    # for i in range(len(labels)):
    #     pos = 0
    #     for j in range(knear):
    #         if ytraining[nearest[j][0]] == labels[i]:
    #            pos += 1
    #     posproba = binDi(pos, knear, labelproba[i])
    #     neg = knear-pos
    #     negproba = binDi(pos, knear, 1-labelproba[i])
    #     if negproba*(1-labelproba[i]) >= posproba*labelproba[i]:
    #         result.append(1)
    #     else:
    #         result.append(0)

    # result.append(pos/knear)


    #     # labelproba = []
    #     # for i in range(len(labels)):
    #     #     labelproba.append(labelcount[i]/len(ytraining))
    #
    #     nearest = {}
    #     for i in range(len(xtrain)):
    #         sqrsum = 0
    #         for j in range(len(x)):
    #             sqrsum = sqrsum + pow((xtrain[i][j]-x[j]), 2)
    #         nearest[i] = sqrt(sqrsum)
    #     nearest = sorted(nearest.items(), key=operator.itemgetter(1))
    #     kprob = [0]*len(labels)
    #     for i in range(knear):
    #         for j in range(len(labels)):
    #             # print(i, knear, nearest[i][0], ytraining[nearest[i][0]], labels[j])
    #             if ytraining[nearest[i][0]] == labels[j]:
    #                 kprob[j] += 1
    #     # print(kprob)
    #     for i in range(len(kprob)):
    #         kprob[i] = kprob[i]/labelcount[i] # *labelproba[i]
    #
    #     result = labels[kprob.index(max(kprob))]
    return result


def Accuracy(predict, accur): # using 0/1 loss
    error = 0
    errors = []
    for i in range(len(predict)):
        for j in range(len(predict[i])):
            if predict[i][j] != accur[i][j]:
                errors.append([predict[i],"should be", accur[i]])
                error +=1
                break
    # print(error, "errors are", errors)
    return 100*(1 - error / len(predict))



# make list of attribute value for x, and list in format [0, 0, 1] for label for y
x = []
y = []
for i in range(len(iris.data)):
    x.append(list(iris.data[i]))
    temp = [0]*len(getLabel(iris.target))
    temp[iris.target[i]] = 1
    y.append(temp)

BRave = []
LPave = []
rakelAve = []
kmeanAve = []

# define the range for the k in knn
firstK = 17
lastK = 17
krange = range(firstK, lastK + 1)

oneK = True # if we already chosed a k

iterations = 10 # number of iterations

if firstK != lastK:
    oneK = False
    for i in range(len(krange)):
        BRave.append([])
        LPave.append([])
        rakelAve.append([])
        kmeanAve.append([])



for r in range(iterations):


    # shuffle the data to have homogene data (avoid having whole folder of same class)
    tempshuffle = list(zip(x, y))
    random.shuffle(tempshuffle)
    x, y = zip(*tempshuffle)



    # knearResultBR = []
    # knearResultLP = []
    # knearResultrakel = []
    # knearResultKmean = []


    # test different k for the knn
    for k in krange:
        print(k)

        # data divided in 10 folder for cross validation
        kfolder = 10
        #
        # to register the accuracy for each iteration of cross-validation
        BRacc = []
        LPacc = []
        rakelacc = []
        Kmeanacc = []

        # for each folder to be taken as test data
        for p in range(kfolder):

            xtrain = []
            xtest = []
            ytrain = []
            ytest = []

            for i in range(len(iris.data)):
                if p*(len(iris.data) / kfolder) <= i < (p + 1)*(len(iris.data) / kfolder):
                    xtest.append(x[i])
                    ytest.append(y[i])
                else:
                    xtrain.append(x[i])
                    ytrain.append(y[i])


            centers = kmean(xtrain, 3)
            discovered = []
            for i in range(len(xtest)):
                discovered.append(kmeancl(centers, xtest[i]))
                temp = [0] * 3
                temp[discovered[i]] = 1
                discovered[i] = temp

            Kmeanacc.append(kmeanMaxAcc(ytest, discovered))

            # Binary relevance
            #  can create one mor label, when classification

            yBR = []
            for i in range(len(ytrain[0])):
                temp = []
                for j in range(len(ytrain)):
                    temp.append(ytrain[j][i])
                yBR.append(temp)
            BRresult = []
            for i in range(len(xtest)):
                temp = []
                for j in range(len(ytrain[0])):
                    temp.append(muCl(xtrain, yBR[j], xtest[i], k))
                BRresult.append(temp)

            BRacc.append(Accuracy(BRresult, ytest))


            # chain classifier ?
            # important if doing some NLP


            # Label powerset


            yLP = []
            for i in range(len(ytrain)):
                temp = ""
                for j in range(len(ytrain[0])):
                    temp = temp + str(ytrain[i][j])
                yLP.append(temp)

            LPresult = []
            for i in range(len(xtest)):
                LPresult.append(muCl(xtrain, yLP, xtest[i], k))
            labels = getLabel(yLP)

            for i in range(len(LPresult)):
                for j in range(len(labels)):
                    if LPresult[i] == labels[j]:
                        temp = []
                        for o in range(len(labels[j])):
                            temp.append(int(labels[j][o]))
                LPresult[i] = temp
            # print(LPresult)
            # print(ytest)
            LPacc.append(Accuracy(ytest, LPresult))


            # rRAkel

            # divide the different subsets of LP
            yrakel = []
            for i in range(len(ytrain[0])):
                yrakel.append([])
            for i in range(len(ytrain)):
                for j in range(len(ytrain[0])):
                    l = 0
                    temp = ""
                    for m in range(len(ytrain[0])-1):
                        if l == j - 1:
                            l +=1
                            temp += str(ytrain[i][l])
                            l +=1
                        else:
                            temp += str(ytrain[i][l])
                            l +=1

                    yrakel[j].append(temp)
            divrakelresult = []
            for i in range(len(yrakel)):
                temp = []
                for j in range(len(xtest)):
                    temp.append(muCl(xtrain, yrakel[i], xtest[j], k))
                divrakelresult.append(temp)

            rakelresults = []

            labels = []
            for i in range(len(yrakel)):
                labels.append(getLabel(yrakel[i]))


            for i in range(len(divrakelresult[0])):
                temp = [0, 0, 0]
                for j in range(len(divrakelresult)):
                    l = 0
                    for m in range(len(labels[0][0])):
                        if l == j-1:
                            l +=1
                            temp[l] = temp[l] + int(divrakelresult[j][i][m])
                            l +=1
                        else:
                            temp[l] = temp[l] + int(divrakelresult[j][i][m])
                            l +=1
                rakelresults.append(temp)

            for i in range(len(rakelresults)):
                for j in range(len(rakelresults[i])):
                    if rakelresults[i][j] > 1:
                        rakelresults[i][j] = 1

            rakelacc.append(Accuracy(ytest, rakelresults))


            # pairwise
            # could be use to decide for a equality or two classes in case of classification

            # if we do 4class proble, same principle that rakel
            #
            # y1v2 = {}
            # y1v3 = {}
            # y2v3 = {}
            #
            # for i in range(len(x)):
            #     if ytrain[i][0] != ytrain[i][1]:
            #         y1v2[i] = ytrain[i][0]
            #     if ytrain[i][0] != ytrain[i][2]:
            #         y1v3[i] = ytrain[i][0]
            #     if ytrain[i][1] != ytrain[i][2]:
            #         y2v3[i] = ytrain[i][1]
            #
            # x1or2 = []
            # y1or2 = []
            # for i in range(len(y1v2)):
            #     x1or2.append(xtrain[list(y1v2)[i]])
            #     y1or2.append(y1v2[i])
            #
            # result1or2 = []
            # for i in range(len(xtest)):
            #     result1or2.append(muCl(x1or2, y1or2, xtest[i]))

            # print(result1or2)



            # copy-weight
            # same than LP with multi classification since weight is always 1 (one label per instance)

            #
            # yCW = []
            # for i in range(len(ytrain)):
            #     temp = []
            #     for j in range(len(ytrain[0])):
            #         if ytrain[i][j] == 1:
            #             temp.append(j)
            #     yCW.append(temp)
            #
            # CWresult = []
            # for i in range():
            #     temp = []
            #     for j in range():
            #         if > 0.5:
            #             temp.append(1)
            #         else:
            #             temp.append(0)
            #     CWresult.append(temp)
            # print(yCW)

        if oneK:
            BRave.append(sum(BRacc) / len(BRacc))
            LPave.append(sum(LPacc) / len(LPacc))
            rakelAve.append(sum(rakelacc) / len(rakelacc))
            kmeanAve.append(sum(Kmeanacc) / len(Kmeanacc))
        else:
            BRave[k-firstK].append(sum(BRacc) / len(BRacc))
            LPave[k-firstK].append(sum(LPacc) / len(LPacc))
            rakelAve[k-firstK].append(sum(rakelacc) / len(rakelacc))
            kmeanAve[k-firstK].append(sum(Kmeanacc) / len(Kmeanacc))

if oneK :
    BRave = sum(BRave) / len(BRave)
    LPave = sum(LPave) / len(LPave)
    rakelAve = sum(rakelAve) / len(rakelAve)
    kmeanAve = sum(kmeanAve) / len(kmeanAve)
    print('K-mean clutering accuracy is', kmeanAve, '%')
    print('Binary relevance accuracy is', BRave, '%')
    print('Label powerset accuracy is', LPave, '%')
    print('Rakel result accuracy is', rakelAve, '%')
else:
    for r in range(len(krange)):
        BRave[r] = sum(BRave[r]) / len(BRave[r])
        LPave[r] = sum(LPave[r]) / len(LPave[r])
        rakelAve[r] = sum(rakelAve[r]) / len(rakelAve[r])
        kmeanAve[r] = sum(kmeanAve[r]) / len(kmeanAve[r])
    xaxe = []
    for i in krange:
        xaxe.append(i)
    plt.plot(xaxe, BRave, 'b', label='Binary relevance')
    plt.plot(xaxe, LPave, 'g', label='Label powerset')
    plt.plot(xaxe, rakelAve, 'r', label='rakel')
    plt.legend()
    plt.show()