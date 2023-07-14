from datetime import datetime, timedelta
from collections import defaultdict
import random
import copy
import numpy as np
import matplotlib.pyplot as plt


flightFile = open("flights.txt", "r")
flightList = flightFile.readlines()
flightFile.close()

individuoDict = {
    "Ida": {
        "LISFCO": [],
        "MADFCO": [],
        "CDGFCO": [],
        "DUBFCO": [],
        "BRUFCO": [],
        "LHRFCO": []
    },
    "Volta": {
        "FCOLIS": [],
        "FCOMAD": [],
        "FCOCDG": [],
        "FCODUB": [],
        "FCOBRU": [],
        "FCOLHR": []
    }
}

population = 1000
mutationRate = 24  # Chance de mutação; quanto maior, mais chance
mateRate = 40  # Chance de cruzamento; quanto maior, mais chance
tournamentConstant = 15  # Número de indivíduos selecionados para o torneio
tournamentSelectionRate = 34  # Chance de selecionar o pior

numOfGenerations = 50

populationList = []
firstTimeRun = True
theBestOne = individuoDict.copy()


def criar_graficos_juntos(listToPlt):
    num_graficos = len(listToPlt[0])

    # Configurar o layout dos subgráficos
    nrows = int(num_graficos / 2) + num_graficos % 2
    ncols = 2

    # Criar a figura e os subgráficos
    fig, axs = plt.subplots(nrows, ncols)
    for valores_list in listToPlt:

        names = ["Best Fitness", "Worst Fitness",
                 "Stardard Deviation", "Average Fitness"]
        # Iterar sobre os valores e criar os gráficos individuais
        for i, valores in enumerate(valores_list):
            row = i // ncols
            col = i % ncols

            indices = range(1, len(valores) + 1)

            axs[row, col].plot(indices, valores)
            axs[row, col].set_xlabel('Geração')
            axs[row, col].set_ylabel('Fitness')
            axs[row, col].set_title(names[i])

        # Ajustar o espaçamento entre os subgráficos
        plt.tight_layout()

    # Exibir os gráficos
    plt.show()


def calcular_desvio_padrao(valores):
    # Converter a lista de valores em um array do NumPy
    array_valores = np.array(valores)

    # Calcular o desvio padrão usando a função std do NumPy
    desvio_padrao = np.std(array_valores)

    return desvio_padrao


def calcular_media(valores):
    # Verificar se a lista está vazia
    if len(valores) == 0:
        return 0

    # Calcular a média
    soma = sum(valores)
    media = soma / len(valores)

    return media


hash_table = defaultdict(list)


def encontrar_maior_menor_hora(lista_horas):
    menor_hora = None
    maior_hora = None

    for hora_str in lista_horas:
        hora_obj = datetime.strptime(hora_str, "%H:%M").time()

        if menor_hora is None or hora_obj < menor_hora:
            menor_hora = hora_obj
        if maior_hora is None or hora_obj > maior_hora:
            maior_hora = hora_obj

    return menor_hora.strftime("%H:%M"), maior_hora.strftime("%H:%M")


def calcular_duracao_viagem(hora_inicio, hora_termino):
  formato = "%H:%M"  # Formato das horas (horas:minutos)
  inicio = datetime.strptime(hora_inicio, formato)
  termino = datetime.strptime(hora_termino, formato)
  # Tratamento para término após a meia-noite
  if termino < inicio:
      termino += timedelta(days=1)  # Adiciona um dia ao horário de término

  duracao = termino - inicio
  duracao_em_minutos = (duracao.seconds) // 60

  return duracao_em_minutos


def calculaFitness(individuo):
    listOfHours = []
    fitness = 0
    for flight in individuo["Ida"]:
        listOfHours.append(individuo["Ida"][flight][3])

    menor, maior = encontrar_maior_menor_hora(listOfHours)

    for flight in individuo["Ida"]:
        value = int(individuo["Ida"][flight][4])
        fitness += calcular_duracao_viagem(
            individuo["Ida"][flight][3], maior) + value

    listOfHours = []
    for flight in individuo["Volta"]:
        listOfHours.append(individuo["Volta"][flight][2])

    menor, maior = encontrar_maior_menor_hora(listOfHours)

    for flight in individuo["Volta"]:
        value = int(individuo["Volta"][flight][4])
        fitness += calcular_duracao_viagem(
            menor, individuo["Volta"][flight][2]) + value

    return fitness


def tournament():
    auxiliarPopulationList = []

    for i in range(population):
        firstTimeRun = True
        tournamentAuxiliarList = []
        theBestLocal = individuoDict.copy()
        theWorstLocal = individuoDict.copy()
        tournamentAuxiliarList = random.sample(
            populationList, tournamentConstant)
        for randomIndividual in tournamentAuxiliarList:

            if firstTimeRun:
                theBestLocal = randomIndividual.copy()
                theWorstLocal = randomIndividual.copy()
                firstTimeRun = False
            else:
                if theBestLocal[1] > randomIndividual[1]:
                    theBestLocal = randomIndividual.copy()

                if theWorstLocal[1] < randomIndividual[1]:
                    theWorstLocal = randomIndividual.copy()

        if random.randint(0, 100) <= int(tournamentSelectionRate):
            auxiliarPopulationList.append(theWorstLocal)

        else:
            auxiliarPopulationList.append(theBestLocal)
    return copy.deepcopy(auxiliarPopulationList)


for i in flightList:
    line = i.split()
    line = (''.join(line)).split(",")

    indice = line[0] + line[1]
    hash_table[indice].append(line)


def crossover(individuo1, individuo2):
    filho1 = individuoDict.copy()
    filho2 = individuoDict.copy()

    for tipo in individuoDict:
        for voo in individuoDict[tipo]:
            if random.random() < mateRate / 100:
                filho1[tipo][voo] = individuo2[tipo][voo]
                filho2[tipo][voo] = individuo1[tipo][voo]
            else:
                filho1[tipo][voo] = individuo1[tipo][voo]
                filho2[tipo][voo] = individuo2[tipo][voo]

    return filho1, filho2


def generate_next_generation(populationList):
    newPopulationList = []

    while len(newPopulationList) < population:
        pai1 = random.choice(populationList)
        populationList.remove(pai1)
        pai2 = random.choice(populationList)
        populationList.remove(pai2)
        if (random.random() < mateRate/100):
            filho1, filho2 = crossover(pai1[0], pai2[0])
        else:
            filho1 = pai1[0]
            filho2 = pai2[0]
        newPopulationList.append(
            [copy.deepcopy(filho1), calculaFitness(filho1)])
        newPopulationList.append(
            [copy.deepcopy(filho2), calculaFitness(filho2)])

    return newPopulationList


def mutate(individuo):
    for tipo in individuo:
        for voo in individuo[tipo]:
            if random.random() < (mutationRate / 100):
                individuo[tipo][voo] = random.choice(hash_table[voo])

    return individuo


def encontrar_10_menores_sem_repeticao(valores):
    menores_unicos = sorted(set(valores))[:10]
    return menores_unicos


def encontrar_individuo_por_valor(lista, valor):
    for individuo in lista:
        if individuo[1] == valor:
            return individuo


plotList = []

for i in range(0, 30):

    for i in range(population):
        individuo = individuoDict.copy()
        for j in individuoDict:
            for flight in individuo[j]:
                individuo[j][flight] = random.choice(hash_table[flight])

        fitness = int(calculaFitness(individuo))
        auxList = []
        auxList.append(individuo)
        auxList.append(fitness)
        if firstTimeRun:
            theBestOne = copy.deepcopy(auxList)
            firstTimeRun = False
        else:
            if theBestOne[1] > auxList[1]:
                theBestOne = copy.deepcopy(auxList)
        populationList.append(copy.deepcopy(auxList))
    elite = []
    primeiroElite = True
    listOfValuesToPlot = [[], [], [], []]
    for generation in range(numOfGenerations):
        firstTimeRun = True
        theBestLocalOne = individuoDict.copy()
        theWorstLocalOne = individuoDict.copy()
        stdDeviation = 0
        Average = 0
        fitnessList = []
        print("------------------------")
        print("Generation:", generation)
        print("Best fitness:", theBestOne[1])
        print("------------------------")

        populationToMateList = tournament()
        populationList = generate_next_generation(populationToMateList)

        for i in range(0, len(populationList)):
            if random.random() < (mutationRate / 100):
                populationList[i][0] = mutate(populationList[i][0])
                populationList[i][1] = calculaFitness(
                    populationList[i][0])

        for individuo in populationList:
            if firstTimeRun:
                theBestLocalOne = individuo.copy()
                theWorstLocalOne = individuo.copy()
                firstTimeRun = False
            else:
                if theBestLocalOne[1] > individuo[1]:
                    theBestLocalOne = individuo.copy()
                if theWorstLocalOne[1] < individuo[1]:
                    theWorstLocalOne = individuo.copy()
            fitnessList.append(individuo[1])
        bestFit = encontrar_10_menores_sem_repeticao(fitnessList)

        # if(primeiroElite):
        #     primeiroElite = False
        #     for i in bestFit:
        #         elite.append(encontrar_individuo_por_valor(populationList,i))
        # else:
        #     numInElite = []
        #     for i in elite:
        #        numInElite.append(i[1])
        #     numeros_entrarao = []
        #     for numero in bestFit:
        #         for i in elite:
        #             if numero < i[1] and numero not in numInElite:

        #                 elite.insert(elite.index(i), encontrar_individuo_por_valor(populationList, numero))

        #                 numeros_entrarao.append(numero)
        #                 break

        if theBestLocalOne[1] < theBestOne[1]:
            theBestOne = theBestLocalOne.copy()

        stdDeviation = calcular_desvio_padrao(fitnessList)
        Average = calcular_media(fitnessList)
        # Elitism
        populationList[random.randint(0, population-1)] = theBestOne
        # randNums = random.sample(range(0,population),2)
        # aux = 0
        # for i in randNums:
        #     populationList[i] = elite[aux]
        #     print(elite[aux][1])
        #     aux+=1
        listOfValuesToPlot[0].append(theBestLocalOne[1])
        listOfValuesToPlot[1].append(theWorstLocalOne[1])
        listOfValuesToPlot[2].append(stdDeviation)
        listOfValuesToPlot[3].append(Average)
        print("Best generation fitness: " + str(theBestLocalOne[1]))
        print("Worst generation fitness: " + str(theWorstLocalOne[1]))
        print("Standard Deviation fitness: " + str(stdDeviation))
        print("Average fitness: " + str(Average))

    listTxtIndividuals = []
    for tipo in theBestOne[0]:
        for flight in theBestOne[0][tipo]:
            individualTxt = ','.join(theBestOne[0][tipo][flight])
            listTxtIndividuals.append(individualTxt)

    print("Best fitness:", theBestOne[1])

    plotList.append(copy.deepcopy(listOfValuesToPlot))
    populationList = []
    firstTimeRun = True
    theBestOne = individuoDict.copy()

criar_graficos_juntos(plotList)
