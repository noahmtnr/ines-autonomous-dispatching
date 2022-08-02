# imports
import sys
import pandas as pd
import numpy as np
from BenchmarkWrapper import BenchmarkWrapper
from Manhattan_Graph_Environment.gym_graphenv.envs.GraphworldManhattan import GraphEnv
import operator
sys.path.insert(0,"")

def main():
    env = GraphEnv(use_config=True)

    benchmark = BenchmarkWrapper(env,"random")
    results = benchmark.read_orders()
    print("Random: ",results)
    benchmark2 = BenchmarkWrapper(env,"cost")
    results = benchmark2.read_orders()
    print("Cost: ",results)
    benchmark3 = BenchmarkWrapper(env,"Rainbow")
    DQNAgent()
    results = benchmark3.read_orders()
    print("Rainbow: ", results)
    benchmark = BenchmarkWrapper(env,"Shares")
    results = benchmark.read_orders()
    print("SharesAgent: ", results)