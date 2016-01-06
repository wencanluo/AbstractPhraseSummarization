import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import fio
import os
import ILP_MC

RougeHead = ['R1-R', 'R1-P', 'R1-F', 'R2-R', 'R2-P', 'R2-F', 'RSU4-R', 'RSU4-P', 'RSU4-F']

myrouges = ['R1-F','R2-F','RSU4-F']

#names = ['BASELINE', '+ MATRIX COMPLETION', '+ CONCEPT WEIGHTING', '+ MC + CW']
names = ['BASELINE-ILP', 'MC', 'CW', 'BOTH']
#colors = ['#238443', "#225ea8", "#cc4c02", "#e31a1c"]
#colors = ['#addd8e', "#7fcdbb", "#fec44f", "#feb24c"]
#colors = ['#f7fcb9', "#edf8b1", "#fff7bc", "#feb24c"]
#colors = ['#f7fcb9', "#efedf5", "#f7fcb9", "#feb24c"]
#colors = ['#f1eef6', "#74a9cf", "#045a8d", "#feb24c"]
#colors = ['#ffffcc', "#a1dab4", "#41b6c4", "#2c7fb8", '#253494']
#colors = ['#f6eff7', "#bdc9e1", "#67a9cf", "#1c9099", '#016c59']
colors = ['#f7f7f7', "#cccccc", "#969696", "#636363", '#252525']

markers = ['o', 's', '^', 'D', '']
hatchs = ['', '///', '\\\\\\', '///\\\\\\', '']
    
#Lengths = [20, 25, 30, 35, 40]
Lengths = [20, 30, 40]

myalpha = 0.9

lambda_range =  np.arange(0.1, 4.1, 0.1)
sparse_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

outputdir = "E:/Dropbox/sharelatex/acl2015_phrase_summarization/"

from mpl_toolkits.mplot3d import Axes3D

def save_pdf(output):
    pp = PdfPages(output)
    plt.savefig(pp, format='pdf')
    pp.close()

def getXYZ(head, body):
    name = head[0]
    
    y = [float(x) for x in head[1:]]
    x = [float(row[0]) for row in body]
    
    m = len(x)
    n = len(y)
    X = np.empty([m, n])
    Y = np.empty([m, n])
    
    for i in range(m):
        for j in range(n):
            X[i][j] = x[i]
            Y[i][j] = y[j]
    
    Z = []
    for row in body:
        Z.append([float(x) for x in row[1:]])
    
    Z = np.array(Z)
    return name, X, Y, Z
    
def plot_3D(head, body):
    name, X, Y, Z = getXYZ(head, body)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('lambda', fontsize=12)
    ax.set_ylabel('sparse ratio', fontsize=12)
    ax.set_zlabel('non zero', fontsize=12)
    
    ax.plot_wireframe(X, Y, Z)
    #plt.show()
    
    pp = PdfPages(outputdir + 'sparse_ratio.pdf')
    plt.savefig(pp, format='pdf')
    pp.close()

def plot_Sparse():
    input = outputdir + 'sparse_ratio.txt'
    head, body = fio.ReadMatrix(outputdir + 'sparse_ratio.txt', hasHead=True)
    
    plot_3D(head, body)
    
def get_Sparse():
    from config import ConfigFile
    
    head = ['lambda'] + sparse_range
    body = []
    
    matrix_dir = "../../data/matrix/exp5/"
    
    for softimpute_lambda in lambda_range:
        row = []
        row.append(softimpute_lambda)
        print softimpute_lambda
        
        for eps in sparse_range:
            #row.append(eps)
            print eps
            
            rank = 500 if softimpute_lambda >= 1.5 else 2000
            
            prefixA = '.' + str(rank) + '_' + str(softimpute_lambda) + '.softA'
            
            row.append(ILP_MC.getSparseRatio(matrix_dir, prefixA=prefixA, eps=eps))
        body.append(row)
    
    fio.WriteMatrix(outputdir + 'sparse_ratio.txt', body, head)

def Get_Baseline():
    ilpdir = "../../data/ILP1_Sentence/"
    
    newname = ilpdir + "rouge.sentence.txt"
    
    head, body = fio.ReadMatrix(newname)
    
    L_index = head.index('L')
    
    X = []
    
    new_body = []
    for row in body:
        if int(row[L_index]) not in Lengths: continue
        
        X.append(float(row[L_index]))
        
        N = len(row)
        score_q3 = [float(x) for x in row[N - len(RougeHead):]]
        score_q2 = [float(x) for x in row[N - len(RougeHead)*2:N - len(RougeHead)]]
        score_q1 = [float(x) for x in row[N - len(RougeHead)*3:N - len(RougeHead)*2]]
        
        ave = (np.array(score_q1) + np.array(score_q2) + np.array(score_q3))/3

        new_row = []
        for rouge in myrouges:
            index = RougeHead.index(rouge)
            new_row.append(ave[index])
        new_body.append(new_row)
    
    Y = []
    
    for i, rouge in enumerate(myrouges):
        Y.append([row[i] for row in new_body])
    
    return X, Y

def Get_Baseline_Split(L = None):
    ilpdir = '../../data/Engineer/ILP_Baseline/'
    
    newname = ilpdir + "rouge.sentence.L30.txt"
    
    head, body = fio.ReadMatrix(newname)
    
    X = []
    
    new_body = [[], [], []]
    for row in body:
        
        X.append(float(row[L_index]))
        
        N = len(row)
        score_q3 = [float(x) for x in row[N - len(RougeHead):]]
        score_q2 = [float(x) for x in row[N - len(RougeHead)*2:N - len(RougeHead)]]
        score_q1 = [float(x) for x in row[N - len(RougeHead)*3:N - len(RougeHead)*2]]
        
        for i, score in enumerate([score_q1, score_q2, score_q3]):
            new_row = []
            for rouge in myrouges:
                index = RougeHead.index(rouge)
                new_row.append(score[index])
            new_body[i].append(new_row)
    
    Y = [[], [], []]
    
    for n in range(3):
        for i, rouge in enumerate(myrouges):
            Y[n].append([row[i] for row in new_body[n]])
    
    Y1 = [y[0][0] for y in Y]
    Y2 = [y[1][0] for y in Y]
    Y3 = [y[2][0] for y in Y]
    
    return X, [Y1, Y2, Y3]

def plot_Baseline():
    X, Y = Get_Baseline()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('L', fontsize=12)
    ax.set_ylabel('Rouge', fontsize=12)
    
    plt.title("baseline")
    plt.grid(True)

    plt.plot(X, Y[0], label=myrouges[0], marker='D', color="b", alpha=0.6, )
    plt.plot(X, Y[1], label=myrouges[1], marker='s', color="r", alpha=0.6, )
    plt.plot(X, Y[2], label=myrouges[2], marker='^', color="g", alpha=0.6, )
    
    #legend = plt.legend(loc='right center', shadow=True, fontsize='x-large')
    legend = plt.legend(loc='right', shadow=True, fontsize='x-large')
    
    #plt.show()
    
    pp = PdfPages(outputdir + 'baseline.pdf')
    plt.savefig(pp, format='pdf')
    pp.close()

def plot_Length_ROUGE():
    X, Y1 = Get_Baseline()
    
    X2, Y2 = get_UnsupervisedMC()
    
    X3, Y3 = get_CW()
    
    #X4, Y4 = get_MC_CW()
    
    assert(X==X2)
    assert(X==X3)
    #assert(X==X4)
    
    Y1 = np.array(Y1)*100
    Y2 = np.array(Y2)*100
    Y3 = np.array(Y3)*100
    
    X = np.array([1, 2, 3])
    
    fontsize = 12
    labelsize = 8
    legendsize = 7.5
    
    fig = plt.figure(figsize=(6*0.9, 2.5*0.9))
    
    ax = fig.add_subplot(121)
    #ax.grid(True)
    
    #ax.set_xlabel('L', fontsize=12)
    ax.set_ylabel('R-1 F-score (%)', fontsize=fontsize)
    
    #alpha = 1.0
    
    #markersize=6
    
    linewidth=2.5
    
    alpha = myalpha
    
    markersize=6
        
    w = 0.7
    ax.set_ylim([10, 45])
    
    ax.bar(X-w/2, Y1[0], width=w/3, label=names[0], hatch=hatchs[0], color=colors[0], alpha=alpha, )
    ax.bar(X-w/2+w/3, Y2[0], width=w/3, label=names[1], hatch=hatchs[1], color=colors[1], alpha=alpha, )
    ax.bar(X-w/2+w/3+w/3, Y3[0], width=w/3, label=names[2], hatch=hatchs[2], color=colors[2], alpha=alpha, )
    
    plt.xticks(np.array(X), ['L=20', 'L=30', 'L=40'])
    plt.tick_params(axis='both', which='major', labelsize=labelsize)
    plt.tick_params(axis='x', which='major', labelsize=labelsize+3)

    plt.tight_layout()
        
    #l0_1 = ax.plot(X, Y1[0], label=names[0], marker=markers[0], markersize=markersize, linewidth=linewidth, color=colors[0], alpha=alpha, )
    #l0_2 = ax.plot(X, Y2[0], label=names[1], marker=markers[1], markersize=markersize, linewidth=linewidth, color=colors[1], alpha=alpha, )
    #l0_3 = ax.plot(X, Y3[0], label=names[2], marker=markers[2], markersize=markersize+2, linewidth=linewidth, color=colors[2], alpha=alpha, )
    #l0_4 = ax.plot(X, Y4[0], label=names[3], marker=markers[3], markersize=markersize, linewidth=linewidth, color=colors[3], alpha=alpha, )
    ax.xaxis.set_ticks(np.array(X))
    ax.legend(loc=0,fancybox=True, prop={'size':legendsize})
    
    ax1 = fig.add_subplot(122)
    #ax1.set_xlabel('L', fontsize=12)
    ax1.set_ylabel('R-2 F-score (%)', fontsize=fontsize)
    #ax1.grid(True)
    ax1.set_ylim([0, 16])
    
    ax1.bar(X-w/2, Y1[1], width=w/3, label=names[0], hatch=hatchs[0], color=colors[0], alpha=alpha, )
    ax1.bar(X-w/2+w/3, Y2[1], width=w/3, label=names[1], hatch=hatchs[1], color=colors[1], alpha=alpha, )
    ax1.bar(X-w/2+w/3+w/3, Y3[1], width=w/3, label=names[2], hatch=hatchs[2], color=colors[2], alpha=alpha, )
    
#     l1_1 = ax1.plot(X, Y1[1], label=names[0], marker=markers[0], markersize=markersize, linewidth=linewidth, color=colors[0], alpha=alpha, )
#     l1_2 = ax1.plot(X, Y2[1], label=names[1], marker=markers[1], markersize=markersize, linewidth=linewidth, color=colors[1], alpha=alpha, )
#     l1_3 = ax1.plot(X, Y3[1], label=names[2], marker=markers[2], markersize=markersize+2, linewidth=linewidth, color=colors[2], alpha=alpha, )
#     #l1_4 = ax1.plot(X, Y4[1], label=names[3], marker=markers[3], markersize=markersize, linewidth=linewidth, color=colors[3], alpha=alpha, )
    ax1.xaxis.set_ticks(np.array(X))
    plt.xticks(np.array(X), ['L=20', 'L=30', 'L=40'])
    plt.tick_params(axis='both', which='major', labelsize=labelsize)
    plt.tick_params(axis='x', which='major', labelsize=labelsize+3)
    
    plt.tight_layout()
    
    ax1.legend(loc=0,fancybox=True, prop={'size':legendsize})
    
    #plt.figlegend((l0, l1), ('BASELINE', 'BASELINE+MC'), 'upper right')
    #legend = plt.legend(loc='right', shadow=True, fontsize='x-large')
    
    fig.subplots_adjust(left=0.08, bottom=0.09, right=0.999, top=0.95)
 
    #plt.show()
     
    pp = PdfPages(outputdir + 'R1_R2_Length.pdf')
    plt.savefig(pp, format='pdf')
    pp.close()

def plot_Length_ROUGE2():
    X, Y1 = Get_Baseline()
    
    X2, Y2 = get_UnsupervisedMC()
    
    X3, Y3 = get_CW()
    
    X4, Y4 = get_MC_CW()
    
    assert(X==X2)
    assert(X==X3)
    assert(X==X4)
    
    fig = plt.figure()
    fig = plt.figure(figsize=(12, 5))
    fig.subplots_adjust(left=0.06, bottom=0.1, right=0.97)

    ax = fig.add_subplot(121)
    ax.grid(True)
    
    ax.set_xlabel('L', fontsize=12)
    ax.set_ylabel('R-1 F-score', fontsize=12)
    
    alpha = 1.0
    
    markersize=6
    
    linewidth=2.5
    
    l0_1 = ax.plot(X, Y1[0], label=names[0], marker=markers[0], markersize=markersize, linewidth=linewidth, color=colors[0], alpha=alpha, )
    l0_2 = ax.plot(X, Y2[0], label=names[1], marker=markers[1], markersize=markersize, linewidth=linewidth, color=colors[1], alpha=alpha, )
    l0_3 = ax.plot(X, Y3[0], label=names[2], marker=markers[2], markersize=markersize+2, linewidth=linewidth, color=colors[2], alpha=alpha, )
    l0_4 = ax.plot(X, Y4[0], label=names[3], marker=markers[3], markersize=markersize, linewidth=linewidth, color=colors[3], alpha=alpha, )
    ax.xaxis.set_ticks(np.array(X))
    ax.legend(loc=0,fancybox=True, prop={'size':11})
    
    ax1 = fig.add_subplot(122)
    ax1.set_xlabel('L', fontsize=12)
    ax1.set_ylabel('R-2 F-score', fontsize=12)
    ax1.grid(True)
    l1_1 = ax1.plot(X, Y1[1], label=names[0], marker=markers[0], markersize=markersize, linewidth=linewidth, color=colors[0], alpha=alpha, )
    l1_2 = ax1.plot(X, Y2[1], label=names[1], marker=markers[1], markersize=markersize, linewidth=linewidth, color=colors[1], alpha=alpha, )
    l1_3 = ax1.plot(X, Y3[1], label=names[2], marker=markers[2], markersize=markersize+2, linewidth=linewidth, color=colors[2], alpha=alpha, )
    l1_4 = ax1.plot(X, Y4[1], label=names[3], marker=markers[3], markersize=markersize, linewidth=linewidth, color=colors[3], alpha=alpha, )
    ax1.xaxis.set_ticks(np.array(X))
    
    ax1.legend(loc=0,fancybox=True, prop={'size':11})
    
    #plt.figlegend((l0, l1), ('BASELINE', 'BASELINE+MC'), 'upper right')
    #legend = plt.legend(loc='right', shadow=True, fontsize='x-large')
     
    #plt.show()
     
    pp = PdfPages(outputdir + 'R1_R2_Length.pdf')
    plt.savefig(pp, format='pdf')
    pp.close()
        
def get_UnsupervisedMC_Lambda():
    ilpdir = "../../data/ILP1_Sentence_MC/"
    
    newname = ilpdir + "rouge.sentence.s0.txt"
    head, body = fio.ReadMatrix(newname)
    
    L_index = head.index('L')
    lambda_index = head.index('lambda')
    
    X = []
    
    new_body = []
    for L in ['30']:
        for row in body:
            if row[L_index] != L: continue
            X.append(float(row[lambda_index]))
            
            N = len(row)
            score_q3 = [float(x) for x in row[N - len(RougeHead):]]
            score_q2 = [float(x) for x in row[N - len(RougeHead)*2:N - len(RougeHead)]]
            score_q1 = [float(x) for x in row[N - len(RougeHead)*3:N - len(RougeHead)*2]]
            
            ave = (np.array(score_q1) + np.array(score_q2) + np.array(score_q3))/3

            new_row = []
            for rouge in myrouges:
                index = RougeHead.index(rouge)
                new_row.append(ave[index])
            new_body.append(new_row)
    Y = []
    
    for i, rouge in enumerate(myrouges):
        Y.append([row[i] for row in new_body])
    
    return X, Y
            
def plot_UnsupervisedMC():
    X, Y = get_UnsupervisedMC_Lambda()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('lambda', fontsize=12)
    ax.set_ylabel('Rouge', fontsize=12)
    
    plt.title("baseline+MC")
    plt.grid(True)

    plt.plot(X, Y[0], label=myrouges[0], marker='D', color="b", alpha=0.6, )
    plt.plot(X, Y[1], label=myrouges[1], marker='s', color="r", alpha=0.6, )
    plt.plot(X, Y[2], label=myrouges[2], marker='^', color="g", alpha=0.6, )
    
    #legend = plt.legend(loc='right center', shadow=True, fontsize='x-large')
    legend = plt.legend(loc='right', shadow=True, fontsize='x-large')
    
    #plt.show()
    
    pp = PdfPages(outputdir + 'baseline+MC.pdf')
    plt.savefig(pp, format='pdf')
    pp.close()

def get_UnsupervisedMC(softimpute_lambda='2.0'):
    ilpdir = "../../data/ILP1_Sentence_MC/"
    
    newname = ilpdir + "rouge.sentence.s0.txt"
    head, body = fio.ReadMatrix(newname)
    
    L_index = head.index('L')
    lambda_index = head.index('lambda')
    
    X = []
    
    new_body = []
    for row in body:
        if int(row[L_index]) not in Lengths: continue
        if (row[lambda_index] != '2.0'): continue
        
        X.append(float(row[L_index]))
        
        N = len(row)
        score_q3 = [float(x) for x in row[N - len(RougeHead):]]
        score_q2 = [float(x) for x in row[N - len(RougeHead)*2:N - len(RougeHead)]]
        score_q1 = [float(x) for x in row[N - len(RougeHead)*3:N - len(RougeHead)*2]]
        
        ave = (np.array(score_q1) + np.array(score_q2) + np.array(score_q3))/3

        new_row = []
        for rouge in myrouges:
            index = RougeHead.index(rouge)
            new_row.append(ave[index])
        new_body.append(new_row)
    
    Y = []
    for i, rouge in enumerate(myrouges):
        Y.append([row[i] for row in new_body])
    
    return X, Y

def get_UnsupervisedMC_Split(softimpute_lambda='2.0', L=None):
    ilpdir = "../../data/ILP1_Sentence_MC/"
    
    newname = ilpdir + "rouge.sentence.s0.txt"
    head, body = fio.ReadMatrix(newname)
    
    L_index = head.index('L')
    lambda_index = head.index('lambda')
    
    X = []
    
    new_body = [[],[],[]]
    for row in body:
        if L != None:
            if row[L_index] != L: continue
        if (row[lambda_index] != '2.0'): continue
        
        X.append(float(row[L_index]))
        
        N = len(row)
        score_q3 = [float(x) for x in row[N - len(RougeHead):]]
        score_q2 = [float(x) for x in row[N - len(RougeHead)*2:N - len(RougeHead)]]
        score_q1 = [float(x) for x in row[N - len(RougeHead)*3:N - len(RougeHead)*2]]
        
        for i, score in enumerate([score_q1, score_q2, score_q3]):
            new_row = []
            for rouge in myrouges:
                index = RougeHead.index(rouge)
                new_row.append(score[index])
            new_body[i].append(new_row)
    
    Y = [[], [], []]
    
    for n in range(3):
        for i, rouge in enumerate(myrouges):
            Y[n].append([row[i] for row in new_body[n]])
    
    Y1 = [y[0][0] for y in Y]
    Y2 = [y[1][0] for y in Y]
    Y3 = [y[2][0] for y in Y]
    
    return X, [Y1, Y2, Y3]

def plot_UnsupervisedMC2():
    X, Y = get_UnsupervisedMC('2.0')
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('L', fontsize=12)
    ax.set_ylabel('Rouge', fontsize=12)
    
    plt.title("baseline+MC")
    plt.grid(True)

    plt.plot(X, Y[0], label=myrouges[0], marker='D', color="b", alpha=0.6, )
    plt.plot(X, Y[1], label=myrouges[1], marker='s', color="r", alpha=0.6, )
    plt.plot(X, Y[2], label=myrouges[2], marker='^', color="g", alpha=0.6, )
    
    #legend = plt.legend(loc='right center', shadow=True, fontsize='x-large')
    legend = plt.legend(loc='right', shadow=True, fontsize='x-large')
    
    #plt.show()
    
    pp = PdfPages(outputdir + 'baseline+CW_L.pdf')
    plt.savefig(pp, format='pdf')
    pp.close()

def get_CW():
    ilpdir = "../../data/ILP_Sentence_Supervised_FeatureWeightingAveragePerceptron/"
    
    newname = ilpdir + "rouge.sentence.txt"
    head, body = fio.ReadMatrix(newname)
    
    L_index = head.index('L')
    
    X = []
    
    new_body = []
    for row in body:
        if int(row[L_index]) not in Lengths: continue
        
        X.append(float(row[L_index]))
        
        N = len(row)
        score_q3 = [float(x) for x in row[N - len(RougeHead):]]
        score_q2 = [float(x) for x in row[N - len(RougeHead)*2:N - len(RougeHead)]]
        score_q1 = [float(x) for x in row[N - len(RougeHead)*3:N - len(RougeHead)*2]]
        
        ave = (np.array(score_q1) + np.array(score_q2) + np.array(score_q3))/3

        new_row = []
        for rouge in myrouges:
            index = RougeHead.index(rouge)
            new_row.append(ave[index])
        new_body.append(new_row)
    
    Y = []
    for i, rouge in enumerate(myrouges):
        Y.append([row[i] for row in new_body])
    
    return X, Y

def get_CW_Split(L=None):
    ilpdir = "../../data/ILP_Sentence_Supervised_FeatureWeightingAveragePerceptron/"
    
    newname = ilpdir + "rouge.sentence.txt"
    head, body = fio.ReadMatrix(newname)
    
    L_index = head.index('L')
    
    X = []
    
    new_body = [[],[],[]]
    for row in body:
        if L != None:
            if row[L_index] != L: continue
        
        if int(row[L_index]) not in Lengths: continue
        
        X.append(float(row[L_index]))
        
        N = len(row)
        score_q3 = [float(x) for x in row[N - len(RougeHead):]]
        score_q2 = [float(x) for x in row[N - len(RougeHead)*2:N - len(RougeHead)]]
        score_q1 = [float(x) for x in row[N - len(RougeHead)*3:N - len(RougeHead)*2]]
        
        for i, score in enumerate([score_q1, score_q2, score_q3]):
            new_row = []
            for rouge in myrouges:
                index = RougeHead.index(rouge)
                new_row.append(score[index])
            new_body[i].append(new_row)
    
    Y = [[], [], []]
    
    for n in range(3):
        for i, rouge in enumerate(myrouges):
            Y[n].append([row[i] for row in new_body[n]])
    
    Y1 = [y[0][0] for y in Y]
    Y2 = [y[1][0] for y in Y]
    Y3 = [y[2][0] for y in Y]
    
    return X, [Y1, Y2, Y3]

def plot_CW():
    X, Y = get_CW()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('L', fontsize=12)
    ax.set_ylabel('Rouge', fontsize=12)
    
    plt.title("baseline+CW")
    plt.grid(True)

    plt.plot(X, Y[0], label=myrouges[0], marker='D', color="b", alpha=0.6, )
    plt.plot(X, Y[1], label=myrouges[1], marker='s', color="r", alpha=0.6, )
    plt.plot(X, Y[2], label=myrouges[2], marker='^', color="g", alpha=0.6, )
    
    #legend = plt.legend(loc='right center', shadow=True, fontsize='x-large')
    legend = plt.legend(loc='right', shadow=True, fontsize='x-large')
    
    #plt.show()
    
    pp = PdfPages(outputdir + 'baseline+CW.pdf')
    plt.savefig(pp, format='pdf')
    pp.close()

def get_MC_CW():
    ilpdir = "../../data/ILP_Sentence_Supervised_FeatureWeightingAveragePerceptronMC/"
    
    newname = ilpdir + "rouge.sentence.txt"
    head, body = fio.ReadMatrix(newname)
    
    L_index = head.index('L')
    
    X = []
    
    new_body = []
    for row in body:
        if int(row[L_index]) not in Lengths: continue
        
        X.append(float(row[L_index]))
        
        N = len(row)
        score_q3 = [float(x) for x in row[N - len(RougeHead):]]
        score_q2 = [float(x) for x in row[N - len(RougeHead)*2:N - len(RougeHead)]]
        score_q1 = [float(x) for x in row[N - len(RougeHead)*3:N - len(RougeHead)*2]]
        
        ave = (np.array(score_q1) + np.array(score_q2) + np.array(score_q3))/3

        new_row = []
        for rouge in myrouges:
            index = RougeHead.index(rouge)
            new_row.append(ave[index])
        new_body.append(new_row)
    
    Y = []
    for i, rouge in enumerate(myrouges):
        Y.append([row[i] for row in new_body])
    
    return X, Y

def get_MC_CW_Split(L=None):
    ilpdir = "../../data/ILP_Sentence_Supervised_FeatureWeightingAveragePerceptronMC/"
    
    newname = ilpdir + "rouge.sentence.txt"
    head, body = fio.ReadMatrix(newname)
    
    L_index = head.index('L')
    
    X = []
    
    new_body = [[],[],[]]
    for row in body:
        if L != None:
            if row[L_index] != L: continue
            
        X.append(float(row[L_index]))
        
        N = len(row)
        score_q3 = [float(x) for x in row[N - len(RougeHead):]]
        score_q2 = [float(x) for x in row[N - len(RougeHead)*2:N - len(RougeHead)]]
        score_q1 = [float(x) for x in row[N - len(RougeHead)*3:N - len(RougeHead)*2]]
        
        for i, score in enumerate([score_q1, score_q2, score_q3]):
            new_row = []
            for rouge in myrouges:
                index = RougeHead.index(rouge)
                new_row.append(score[index])
            new_body[i].append(new_row)
    
    Y = [[], [], []]
    
    for n in range(3):
        for i, rouge in enumerate(myrouges):
            Y[n].append([row[i] for row in new_body[n]])
    
    Y1 = [y[0][0] for y in Y]
    Y2 = [y[1][0] for y in Y]
    Y3 = [y[2][0] for y in Y]
    
    return X, [Y1, Y2, Y3]

def plot_MC_CW():
    X, Y = get_MC_CW()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('L', fontsize=12)
    ax.set_ylabel('Rouge', fontsize=12)
    
    plt.title("baseline+CW+MC")
    plt.grid(True)

    plt.plot(X, Y[0], label=myrouges[0], marker='D', color="b", alpha=0.6, )
    plt.plot(X, Y[1], label=myrouges[1], marker='s', color="r", alpha=0.6, )
    plt.plot(X, Y[2], label=myrouges[2], marker='^', color="g", alpha=0.6, )
    
    #legend = plt.legend(loc='right center', shadow=True, fontsize='x-large')
    legend = plt.legend(loc='right', shadow=True, fontsize='x-large')
    
    #plt.show()
    
    pp = PdfPages(outputdir + 'baseline+CW+MC.pdf')
    plt.savefig(pp, format='pdf')
    pp.close()

def get_rouge_split(input):
    head, body = fio.ReadMatrix(input)
    
    Y = []
    
    for rouge in myrouges:
        index = head.index(rouge)
                
        values = [[],[],[]]
        for i, row in enumerate(body[0:-1]):
            values[i%3].append(float(row[index]))
        
        Y0 = []
        for i in range(3):
            Y0.append(np.average(values[i]))
            
        Y.append(Y0)
    
    return Y

def plot_Split():
    fontsize = 12
    labelsize = 12
    legendsize = 10
    alpha = myalpha
    markersize=6
    w = 0.5
    
    folders = [('MEAD','../../data/Engineer/Mead/'),
               ('LEXRANK','../../data/Engineer/LexRank/'),
           ('SUMBASIC','../../data/Engineer/SumBasic/'),
           ('ILP-BASELINE','../../data/Engineer/ILP_Baseline/'),
           ('ILP-IMPUTATION','../../data/Engineer/ILP_MC/'),
           ]
    
    Ys = []
    for name, folder in folders:
        Y = get_rouge_split(os.path.join(folder, 'rouge.sentence.L30.txt'))
        Y = np.array(Y)*100
        
        Ys.append(Y)
    
    Y = Ys
    
    X = np.array([1, 2, 3])
    
    score_index = 1
    
    fig = plt.figure(figsize=(6*0.9, 2.5*0.9))
    
    ax = fig.add_subplot(111)
    #ax.grid(True)
    
    #ax.set_xlabel('PROMPT', fontsize=12)
    ax.set_ylabel('R-2 F-score (%)', fontsize=fontsize)
    
    
    #ax.set_ylim([10, 56])
    ax.set_ylim([0, 25])
    
    ax.bar(X-w/2-w/3, Y[0][score_index], width=w/3, label=folders[0][0], hatch=hatchs[0], color=colors[0], alpha=alpha, )
    ax.bar(X-w/2, Y[1][score_index], width=w/3, label=folders[1][0], hatch=hatchs[1], color=colors[1], alpha=alpha, )
    ax.bar(X-w/2+w/3, Y[2][score_index], width=w/3, label=folders[2][0], hatch=hatchs[2], color=colors[2], alpha=alpha, )
    ax.bar(X-w/2+w/3+w/3, Y[3][score_index], width=w/3, label=folders[3][0], hatch=hatchs[3], color=colors[3], alpha=alpha, )
    ax.bar(X-w/2+w/3+w/3+w/3, Y[4][score_index], width=w/3, label=folders[4][0], hatch=hatchs[4], color=colors[4], alpha=alpha, )
    
    plt.xticks(np.array(X), ['POI', 'MP', 'LP'])
    plt.tick_params(axis='both', which='major', labelsize=labelsize)
    plt.tick_params(axis='x', which='major', labelsize=labelsize+3)
    
    plt.tight_layout()
    
    #ax.xaxis.set_ticks(np.array(X), ['POI', 'MP', 'LP'])
    ax.legend(loc=0,fancybox=True, prop={'size':legendsize})
    
    
#     ax1 = fig.add_subplot(122)
#     #ax1.set_xlabel('L', fontsize=12)
#     ax1.set_ylabel('R-2 F-score (%)', fontsize=fontsize)
#     #ax1.grid(True)
#     
#     ax1.bar(X-w/2, Y[0][1], width=w/3, label=names[0], hatch=hatchs[0], color=colors[0], alpha=alpha, )
#     ax1.bar(X-w/2+w/3, Y[1][1], width=w/3, label=names[1], hatch=hatchs[1], color=colors[1], alpha=alpha, )
#     ax1.bar(X-w/2+w/3+w/3, Y[2][1], width=w/3, label=names[2], hatch=hatchs[2], color=colors[2], alpha=alpha, )
#     #ax1.bar(X-w/2+w/4+w/4+w/4, Y4[1], width=w/4, label=names[3], hatch=hatchs[3], color=colors[3], alpha=alpha, )
#     
#     ax1.set_ylim([0, 25])
#     
#     ax1.xaxis.set_ticks(np.array(X))
#     
#     ax1.legend(loc=0, fancybox=True, prop={'size':legendsize})
#     plt.tight_layout()
#     
#     #plt.figlegend((l0, l1), ('BASELINE', 'BASELINE+MC'), 'upper right')
#     #legend = plt.legend(loc='right', shadow=True, fontsize='x-large')
#     plt.xticks(np.array(X), ['POI', 'MP', 'LP'])
#     plt.tick_params(axis='both', which='major', labelsize=labelsize)
#     plt.tick_params(axis='x', which='major', labelsize=labelsize+3)
    
    fig.subplots_adjust(left=0, bottom=0.09, right=0.999, top=0.95)

    #plt.show()
    
    pp = PdfPages(outputdir + 'split.pdf')
    plt.savefig(pp, format='pdf')
    pp.close()
    
#     from matplotlib.font_manager import FontProperties
#     fontP = FontProperties()
#     fontP.set_size('small')
#    
#     plt.title("prompts")
#     legend = plt.legend(loc='top right', shadow=True, prop=fontP)
            
    
if __name__ == '__main__':
    datadir = "../../data/"
    
    plot_Split()
    #plot_Length_ROUGE()
    
    #get_Sparse()
    #plot_Sparse()
    #plot_UnsupervisedMC2()
    
#     plot_CW()
#     plot_MC_CW()
    
    print "done"