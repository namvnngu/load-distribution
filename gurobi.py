import pandas as pd
from gurobipy import *


def Solve(p):
    model = Model()             #Initializing a model
    obj = 0                     #The objective of the MIP
    x = []                      #Variable [0,infinite]
    I = len(p)                  #number of node in the instance
    J = len(p[0])               #number of jobs in the instance

    #Initializing the variables within the MIP
    print('Adding variables ...')

    for i in range(I): 
        jTemp = []
        for j in range(J):
            jTemp.append(model.addVar(vtype=GRB.BINARY, name="x_"+str(i+1)+"_"+str(j+1)))

        x.append(jTemp)

    #Initialize a variable for the max makespan
    Cmax = model.addVar(vtype=GRB.INTEGER, name="Cmax")

    model.update()

    #Initializing the objective within the MIP
    print('Setting Objective ...')

    obj = Cmax
    model.setObjective(obj, GRB.MINIMIZE)
    model.update()
    
    #Initializing the constraints within the MIP
    
    #Adding constraints 1, setting up the maximum limit of the makespan
    print('Adding constraint 1')
    for i in range(I):
        constraint = 0
        
        for j in range(J):
            constraint += x[i][j]*p[i][j]

        model.addConstr(constraint <= Cmax, "")

    #Adding contraints 2, ensuring at that all job must be completed 
    #and each job is completed once and only once 
    print('Adding constraint 2')
    for j in range(J):
        constraint = 0

        for i in range(I):
            constraint += x[i][j]

        model.addConstr(constraint == 1, "")

    #Adding constraints 3, forces the decision variables to be binary values
    #These no need to this constraint as the the initialization of the decision variable is already set to GRB.BINARY

    model.update()
    
    print('Optimizing ...')
    model.optimize()
    model.printAttr('X')
    
    #different output for the soluion
    #print('Writing file: solution ...')
    #model.write('out.lp')
    #model.write('out.sol')
    #model.write('out.json')

#read int the instance data
def readInProcessTime():
    df = pd.read_csv("loadDistribution.csv", index_col=0)
    df1 = df.T
    processTime=[]

    for column in df1.columns:
        li = df1[column].tolist()
        processTime.append(li)
    
    return processTime

#main code
p = readInProcessTime()
Solve(p)