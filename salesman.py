
import numpy as np
import operator
import pandas as pd
import random
import matplotlib.pyplot as plt


class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        return np.sqrt((xDis ** 2) + (yDis ** 2))

    def __repr__(self):syna
        return "(" + str(self.x) + "," + str(self.y) + ")"


class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0

    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            # rolling through a group of cities
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance

    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1/ float(self.routeDistance())
        return self.fitness

# initial population

def createRoute(cityList):
    return random.sample(cityList, len(cityList))

#First population (list of routes)

def initialPopulation(popSize, cityList):
    population = []
    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population

def rankRoutes(population):
    fitnesResults = []
    for i in range(0, len(population)):
        fitnesResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnesResults.items(), key = operator.itemgetter(1), reverse = True)

def selection(popRank, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRank), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

    for i in range(0, eliteSize):
        selectionResults.append(popRank[i][0])
    for i in range(0, len(popRank - eliteSize)):
        pick = 100 * random.random()
        for i in range(0, len(popRank)):
            if pick <= df.iat(i, 3):
                selectionResults.append(popRank[i][0])
                break
    return selectionResults

def matingPool(population, selectionRerult):
    matingPool = []
    for i in range(0, len(selectionRerult)):
        index = selectionRerult[i]
        matingPool.append(population[index])
    return matingPool

def breed(father, mother):
    child = []
    childF = []
    childM = []

    geneA = int(random.random() * len(father))
    geneB = int(random.random() * len(father))

    startGene = min(geneA,geneB)
    endGene = max(geneA,geneB)

    for i in range(startGene, endGene):
        childF.append(father[i])

    childM = [item for item in mother if item not in childF]

    child = childF + childM
    return child

def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool,len(matingpool))

    for i in range(0, eliteSize):
        children.append(matingpool[i])

    for i in range(0, length):
        child = breed(pool[i], pool(len(matingpool) - i - 1))
        children.append(child)

    return children

def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapWidth = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWidth]

            individual[swapped] = city2
            individual[swapWidth] = city1

    return individual

def mutatePopulation(population, mutationRate):
    mutationPop = []

    for ind in range(0, len(population)):
        mutationInd = mutate(population[ind], mutationRate)
        mutationPop.append(mutationInd)
    return mutationPop


def nextGeneration(currentGen, elitesite, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, elitesite)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, mutationRate)
    nextgeneration = mutatePopulation(children,mutationRate)
    return nextgeneration

def geneticAlgorithm(population, popSize, elitSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print("initial distance: " + str(1 / rankRoutes(pop)[0][1]))
    for i in range(0, generations):
        pop = nextGeneration(pop, elitSize, mutationRate)

    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    return pop[bestRouteIndex]

cityList = []

for i in range(0, 25):
    cityList.append(City(x=int(random.random() * 200), y=int(random.random() * 200)))

geneticAlgorithm(population=cityList, popSize=100, elitSize=20, mutationRate=0.01,generations=500)
