import os
os.environ['OMP_NUM_THREADS'] = '2'
from itertools import count
import numpy as np
import time
import copy
import sys
import random
import matplotlib.pyplot as plt
import pickle
import numpy as np
from utils import *

'''
本赛题旨在引领选手探索，在NISQ时代规模有限的量子计算机上，求解真实场景中的大规模图分割问题。
本代码为求解最大割
'''
import resource
soft, hard = resource.getrlimit(resource.RLIMIT_AS)
resource.setrlimit(resource.RLIMIT_AS, (1024*1024*4000, hard)) #4G
N_sub=15 # 量子比特的规模限制

filelist=['./graphs/regular_d3_n40_cut54_0.txt', './graphs/regular_d3_n80_cut108_0.txt',
          './graphs/weight_p0.5_n40_cut238.txt','./graphs/weight_p0.2_cut406.txt',
          './graphs/partition_n80_cut226_0.txt', './graphs/partition_n80_cut231_1.txt'
          ]


def build_sub_qubo(solution,N_sub,J,h=None,C=0.0):
    '''
    自定义函数选取subqubo子问题。
    例如可简单选择对cost影响最大的前N_sub个变量组合成为subqubo子问题。
    【注】本函数非必须，仅供参考
    
    返回
    subqubo问题变量的指标，和对应的J，h，C。
    '''
    delta_L=[]
    for j in range(len(solution)):
        copy_x=copy.deepcopy(solution)
        copy_x[j]=-copy_x[j]
        x_flip=copy_x
        sol_qubo=calc_qubo_x(J,solution,h=h,C=C)
        x_flip_qubo=calc_qubo_x(J,x_flip,h=h,C=C)
        delta=x_flip_qubo-sol_qubo
        delta_L.append(delta)
    delta_L=np.array(delta_L)
    #print(delta_L)
    sub_index = np.argpartition(delta_L, -N_sub)[-N_sub:] # subqubo子问题的变量们
    #print('subindex:',sub_index)
    J_sub,h_sub,C_sub = calc_subqubo(sub_index, solution, J, h=h,C=C )
    return sub_index,J_sub,h_sub,C_sub


def solve(sol, g, G):
    '''
    算法总体思路分为两部分：
    第一部分：利用贪心算法寻找初始基态。
    第二部分：基于15比特的量子退火算法，每次选取15比特利用QAOA算法计算基态，然后逐步退火选取最佳结果。
            改进思路：修改降温速度;使用n比特的QAOA算法，n随温度改变，节省时间。
    
    注：若本代码或者思路对你有用，请予以致谢。
    '''
    
    #第一部分
    # 以贪心的方式初始化solution
    n = len(g.nodes)
    unsolve_index = np.arange(0, n, 1)
    solution = np.zeros(n)

    #找到连接节点数目最多的
    max_connet = 0
    for i in unsolve_index:
        if g.degree(i) >= max_connet:
            index = i
            max_connet = g.degree(i)

    solution[index] = 1
    unsolve_index = np.delete(unsolve_index, np.where(unsolve_index==index))
    candidate_index = [index]

    # 以最多节点为开始，将其相邻节点选择为另一种状态
    while len(candidate_index) > 0:
        new_candidate = []
        for up_index in candidate_index:
            for index in [*g[up_index]]:
                if solution[index] == 0:
                    solution[index] = -solution[up_index]
                    unsolve_index = np.delete(unsolve_index, np.where(unsolve_index==index))
                    new_candidate.append(index)
        candidate_index = new_candidate

        # 针对稀疏图重新寻找其它起始点
        if len(new_candidate) == 0 and len(np.where(solution==0)[0]) != 0:
            max_connet = 0
            for i in unsolve_index:
                if g.degree(i) >= max_connet:
                    index = i
                    max_connet = g.degree(i)
            candidate_index = [index]

    solution = (solution+1)/2
    sol = solution
    

    # 第二部分
    # 基于15比特的量子退火算法
    best_score = calc_cut_x(G, sol)
    best_sol = sol
    n = len(g.nodes)
    index = np.arange(0, n, 1)
    for beta in np.linspace(0.5, 1.0, 5):
        for _ in range(15):
            score = calc_cut_x(G, sol)
            sub_index = np.random.choice(index, size=N_sub)
            J_sub,h_sub,C_sub = calc_subqubo(sub_index, sol, G, h=None,C=0.0 )
            new_sol = solve_QAOA(J_sub,h_sub,C_sub,sub_index,sol,depth=3,tol=1e-5)
            new_score = calc_cut_x(G, new_sol)
            #print(new_score, best_score)
            if new_score > best_score: 
                best_sol = new_sol
                best_score = new_score
                #print(best_sol, best_score)
            if new_score > score:
                sol = new_sol
            elif np.exp((new_score-score)*beta)> np.random.random():
                sol = new_sol
    return best_sol, best_score



def run():
    """
    Main run function, for each graph need to run for 20 times to get the mean result.
    Please do not change this function, we use this function to score your algorithm. 
    """
    cut_list=[]
    for filename in filelist[:]:
        #print(f'--------- File: {filename}--------')
        g, G=read_graph(filename)
        n=len(g.nodes) # 图整体规模
        cuts=[]

        #print(f'------turn {turn}------')
        sol=init_solution(n) # 随机初始化解   
        qubo_start = calc_qubo_x(G, sol)
        cut_start =calc_cut_x(G, sol)
        #print('origin qubo:',qubo_start,'|cut:',cut_start)
        solution,cut=solve(sol, g, G) #主要求解函数, 主要代码实现
        cuts.append(cut)
        cut_list.append(cuts)
        
    return np.array(cut_list)  



if __name__== "__main__": 
    #计算分数
    cut_list=run()
    print(cut_list)
    max_arr=np.array([54,108,238,406,226,231])
    size_arr=np.array([40,80,40,80,80,80])
    score=np.array(np.mean(cut_list,axis=1))/max_arr*size_arr
    print('score:',np.sum(score))
    
    