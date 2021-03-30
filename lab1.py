import monkdata as m
import dtree as d
import numpy as np
import matplotlib.pyplot as plt
import drawtree_qt5 as dt5
import random
import statistics

#assignment 1
entropy1 = d.entropy(m.monk1)
entropy2 = d.entropy(m.monk2)
entropy3 = d.entropy(m.monk3)
print('entropy monk1='+str(entropy1))
print('entropy monk2='+str(entropy2))
print('entropy monk3='+str(entropy3))

#assignment 2
N_points = 1000
n_bins = 20
#plt.figure(1)
fig, axs = plt.subplots(1, 2, tight_layout=True)
#high entropy
s = np.random.uniform(0, 1, N_points)
h = axs[0].hist(s, bins=n_bins)
entropy = 0
for v in h[0]:
    if v == 0:
        continue
    en1 = -v/N_points * d.log2(v/N_points)
    entropy = entropy + en1
axs[0].set_title("entropy="+str(entropy))
print('entropy for uniform distribution random sampling='+str(entropy))

#low entropy
s = np.random.randn(N_points)
h = axs[1].hist(s, bins=n_bins)
entropy = 0
for v in h[0]:
    if v == 0:
        continue
    en1 = -v/N_points * d.log2(v/N_points)
    entropy = entropy + en1
axs[1].set_title("entropy="+str(entropy))
print('entropy for normal distribution random sampling='+str(entropy))

#assignment 3
aveGain = {'monk1': {}, 'monk2': {}, 'monk3': {}}
i = 1
j = 1
for monk in [m.monk1, m.monk2, m.monk3]:
    for attribute in m.attributes:
        monkN = 'monk' + str(i)
        attributeN = 'a' + str(j)
        attrN = aveGain[monkN]
        attrN[attributeN] = d.averageGain(monk, attribute)
        aveGain[monkN] = attrN
        j = j+1
    i = i+1
    j = 1

print(aveGain)

#assignment 5
#monk1: a5 is selected
monk1_1 = d.select(m.monk1, m.attributes[4], 1)
monk1_2 = d.select(m.monk1, m.attributes[4], 2)
monk1_3 = d.select(m.monk1, m.attributes[4], 3)
monk1_4 = d.select(m.monk1, m.attributes[4], 4)

aveGain1 = {'monk1_1': {}, 'monk1_2': {}, 'monk1_3': {}, 'monk1_4': {}}
i = 1
j = 1
for monk in [monk1_1, monk1_2, monk1_3, monk1_4]:
    for attribute in m.attributes:
        monkN = 'monk1_' + str(i)
        attributeN = 'a' + str(j)
        attrN = aveGain1[monkN]
        attrN[attributeN] = d.averageGain(monk, attribute)
        aveGain1[monkN] = attrN
        j = j+1
    i = i+1
    j = 1

print(aveGain1)

#branch 1: N/A
#branch 2: a4
#branch 3: a6
#branch 4: a1
monkData = {'monk1_2': {'data': monk1_2, 'branch': 3}, 'monk1_3': {'data': monk1_3, 'branch': 5}, 'monk1_4': {'data':
                    monk1_4, 'branch': 0}}

monk1_maj = d.mostCommon(m.monk1)
monk1_1_maj = d.mostCommon(monk1_1)
print('majority class for monk1: ' + str(monk1_maj))
print('majority class for monk1_1: ' + str(monk1_1_maj))

for mo in monkData:
    moName = mo
    attrNo = monkData[mo]['branch']
    monk_maj = d.mostCommon(monkData[mo]['data'])
    print('majority class for ' + mo + ': ' + str(monk_maj))
    for attrVal in m.attributes[attrNo].values:
        monkTmp = d.select(monkData[mo]['data'], m.attributes[attrNo], attrVal)
        monk_maj = d.mostCommon(monkTmp)
        print('majority class for ' + mo + ', partition ' + str(attrVal) + ': ' + str(monk_maj))


#tree1 = d.buildTree(m.monk1, m.attributes, 2)
#dt5.drawTree(tree1)

t1 = d.buildTree(m.monk1, m.attributes)
print("Training data Monk1", 1-d.check(t1, m.monk1))
print("Testing data Monk1", 1-d.check(t1, m.monk1test))

t2 = d.buildTree(m.monk2, m.attributes)
print("Training data Monk2", 1-d.check(t2, m.monk2))
print("Testing data Monk2", 1-d.check(t2, m.monk2test))

t3 = d.buildTree(m.monk3, m.attributes)
print("Training data Monk3", 1-d.check(t3, m.monk3))
print("Testing data Monk3", 1-d.check(t3, m.monk3test))

#tree2 = d.buildTree(m.monk1, m.attributes)
#dt5.drawTree(tree2)


#assignment6
def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)

    return ldata[:breakPoint], ldata[breakPoint:]

def pruneTree(tree, oldError, validationTree) :
    pruningErrors = list()
    prunedTrees = d.allPruned(tree)

    for trees in prunedTrees:
        pruningErrors.append(d.check(trees,validationTree))
    bestError=max(pruningErrors)
    index_of_bE=pruningErrors.index(bestError)

    #stop pruning when all the pruned trees perform worse than the current candidate
    if bestError < oldError:
        return tree, oldError
    else:
        return pruneTree(prunedTrees[index_of_bE], bestError, validationTree)

# Initialize lists of trees to be pruned

fractions = list((0.3, 0.4, 0.5, 0.6, 0.7, 0.8))

monk1Trees = list()
for i in range(0, len(fractions)) :
    monk1Trees.append(list())

monk1ValTrees = list()
for i in range(len(fractions)) :
    monk1ValTrees.append(list())

monk3Trees = list()
for i in range(len(fractions)) :
    monk3Trees.append(list())

monk3ValTrees = list()
for i in range(len(fractions)) :
    monk3ValTrees.append(list())

# Randomize training and validation sets

for i in range(len(fractions)) :
    for j in range(128) :
        monk1train, monk1val = partition(m.monk1, fractions[i])
        monk3train, monk3val = partition(m.monk3, fractions[i])

        monk1Trees[i].append(d.buildTree(monk1train, m.attributes))
        monk1ValTrees[i].append(monk1val)

        monk3Trees[i].append(d.buildTree(monk3train, m.attributes))
        monk3ValTrees[i].append(monk3val)

Array_1 = np.zeros((6,128))
Array_3 = np.zeros((6,128))

txt = open("as6_data.txt", "w")
txt.write("monk1\n")

for i in range(len(monk1Trees)) :
    txt.write(str("\n\n" + "Fraction: " + str(fractions[i]) + "\n\n"))
    for j in range(len(monk1Trees[i])) :
        prunedTree, prunedError = pruneTree(monk1Trees[i][j], d.check(monk1Trees[i][j], monk1ValTrees[i][j]), monk1ValTrees[i][j])
        Array_1[i][j] = prunedError
        txt.write(str(prunedError) + "\n")
txt.write("Finished monk1")

txt.write("\n\n")

txt.write("monk3\n")
for i in range(len(monk3Trees)) :
    txt.write(str("\n\n" + "Fraction: " + str(fractions[i]) + "\n\n"))
    for j in range(len(monk3Trees[i])) :
        prunedTree, prunedError = pruneTree(monk3Trees[i][j], d.check(monk3Trees[i][j], monk3ValTrees[i][j]), monk3ValTrees[i][j])
        Array_3[i][j] = prunedError
        txt.write(str(prunedError) + "\n")
txt.write("Finished monk3")

x1 = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
y1 = list()
for i in range(6):
    y1.append(np.mean(Array_1[i]))
plot1 = plt.figure(2)
plt.plot(x1, y1)
plt.title('Mean of validation performance on pruned tree, monk1')
plt.xlabel('Fraction of Training Data')
plt.ylabel('Validation Data Mean Performance')

x2 = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
y2 = list()
for i in range(6):
    y2.append(statistics.variance(Array_1[i]))
plot2 = plt.figure(3)
plt.plot(x2, y2)
plt.title('Variance of validation performance on pruned tree, monk1')
plt.xlabel('Fraction of Training Data')
plt.ylabel('Validation Data Performance Variance')

x3 = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
y3 = list()
for i in range(6):
    y3.append(np.mean(Array_3[i]))
plot3 = plt.figure(4)
plt.plot(x3, y3)
plt.title('Mean of validation performance on pruned tree, monk3')
plt.xlabel('Fraction of Training Data')
plt.ylabel('Validation Data Mean Performance')

x4 = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
y4 = list()
for i in range(6):
    y4.append(statistics.variance(Array_3[i]))
plot4 = plt.figure(5)
plt.plot(x4, y4)
plt.title('Variance of validation performance on pruned tree, monk3')
plt.xlabel('Fraction of Training Data')
plt.ylabel('Validation Data Performance Variance')
plt.show()
