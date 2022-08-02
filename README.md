# InES Team Project SS22 - Self-Organising Dispatching System

## Authors:

Agnieszka Lenart, alenart@mail.uni-mannheim.de

Maren Ehrgott, mehrgott@mail.uni-mannheim.de

Denisa Dragota, denisa.dragota@stud.ubbcluj.ro

Cosmina Ionut, cosmina.ionut@stud.ubbcluj.ro

Noah Mautner, nmautner@mail.uni-mannheim.de

Lukas Kirchdorfer, lkirchdo@mail.uni-mannheim.de


University of Mannheim, August 7th, 2022


## Intoduction

### Problem Definition

In today's delivery systems, customers order at a store and the store then uses its own delivery service or an external delivery provider (e.g., Lieferando for food delivery) to deliver the order within a certain deadline to the customer.
	While these delivery services provide a fast and reliable delivery for the customers, they come with multiple issues. They cause a high amount of commercial trips, lead to more vehicles on the roads and require parking space, which is spares in big cities. The deliveries also increase environmental pollution due to their emissions. For small local stores, using an own or external delivery provider might also proof to be too expensive, but not offering delivery services could be critical to the store's success as fast online trade is becoming more and more important.
	In order to overcome these problems, an alternative to traditional delivery systems is needed.

### Goals

In our project, we aim to establish a delivery system that consists of a decentralized dispatching of hitchhiking boxes containing customer orders. The boxes are to be delivered from any store to any customer location (only given these start and final coordinates) within a certain time period within New York City. For this, multi-hop ride sharing (i.e., a box rides on vehicle for some time and then transfers to another ride) using available taxi rides provided by the city of New York is to be implemented. The system is to be trained using historical training data and tested on random and specific custom orders.
	This approach is supposed to satisfy society's mobility and logistics needs (e.g., high demand and lower costs for a ride), challenge the traditional work organization (e.g., saving money for external delivery providers) and improve environmental protection as well as urban quality of life (e.g., less traffic).
  
  ### Methodology
  
  As New York City is to be the setting of the hitchhike system, we constructed the street network of the city as a graph with nodes (location consisting of latitude, longitude and a node ID) and edges (including speed limits and lengths of edges). In this graph we strategically place hubs according to store locations, customer population distribution and the mostly travelled nodes by the taxis. The graph provides the environment for delivering our boxes. Start and final location of delivery are put in as coordinates, which are mapped to the nearest node, which then is mapped to the nearest hub. At the current status, we only provide hub-to-hub delivery, which is why this mapping needs to be done.
	Having the mapping of start and final position on the graph, we initialize the time with the current time and keep track of the deadline (24 hours) to which the box has to be delivered, otherwise the delivery is conceived as failed.
	Having the current position and time as input, we aim to push the box into the direction of the final hub by only taking available shared rides. The available rides are provided from the historical taxi trip data of the city of New York, which was pre-processed and saved into a database. The database is efficiently accessed via SQL views and queries to get the available trips (and respective timestamps).
	Having access to the available trips at a certain hub at a certain time, the box autonomously decides whether to wait, take a trip to some hub, or, in case the deadline is only 2 hours away, book an own trip directly to the final hub. For this, the box is implemented as a Reinforcement Learning Agent (more on that in section "Term Definitions").
	It is trained on the historical trip data and its performance in this training is measured with multiple metrics (more on that in section "Instructions for Training"). In order to finally test the performance on new random and specific custom orders (which we generate), an agent can be compared with benchmarks and other RL agents regarding multiple metrics and the agent's performance and routes taken are visually displayed on a dashboard (more on that in section "Instructions for Testing").
  
## Foundation/Pre-Work  
This section delineates basic terms and principles, as well as a tutorial for installing all required libraries and systems. It serves to support the understanding of the following sections and the execution of the system.

### Term Definitions

### Tutorials for Required Installations


## Repository Structure


### Environment 

### RL Algorithms

### Visualization

### APIs

### Training

### Testing

### Other Files

gitignore, yaml-files

## Environment and Reinforcement Learning
### whatever for environment
### Observation Space
### Action Space
### Agents 
DQN, PPO, Rainbow

## Instruction for Training

### Parameters

	agents,Iterations, hidden layers, activation function, number of steps, ...

### Execution of Training

	- takes about 5 hours for 100 iterations, depends on number of hubs and deadline used


### Results of Training

	@ Maren \\
	
	The orders trained on, as well as the actions and corresponding routes can be found in log-files in the following path: HIER FILEPATH. 
	
	Multiple WandB metrics are used to measure the training performance: 
	Files: GraphWorldManhattan and train[...],
	Output: HIER LINK ZU WANDB. \\
**Available and Useful Trips.** Available equals the shared trips that were available to an agent in one run.
	Available useful equals the useful shared trips available. Useful means that taking the respective trip reduces the remaining distance to the final hub.
	*ratio_shared_available_to_all_steps* Ratio of the number of steps where any kind of shared trip is available to the number of steps in total. Shows how often shared trips are possible. Reflects the sparseness of trips over time.\\
	*shared\_available_useful_to_shared_available* Ratio of the number of steps where useful trips are available ...
  
  
  
  **Reward**
  
  
  **Bookowns, Shares and Waits.**



**Delivered and Not Deliver.**


**Distance Reduced.**



### Instructions for Testing

###Agents
###Benchamrks
###Results of Testing

**Metrics and Ranking**

**Dashboards and Visualisation**



## Package Installation process:

Donwload Anaconda: https://www.anaconda.com/products/individual

1. Install Python version 3.9 in anaconda prompt

- $ conda install python = 3.9

3. create new environment

- open anaconda prompt
- $ conda config --prepend channels conda-forge
- $ conda create -n teamproject --strict-channel-priority osmnx
- $ conda activate teamproject
- open anaconda navigator, select environment teamproject in the drop down menu and install Jupyter Notebook

4. Install pytorch within teamproject environment

- $ pip3 install torch torchvision torchaudio

5. Install open AI gym

- $ pip install pygame
- $ pip install gym

6. Install RLlib within teamproject env

- $ pip install -U ray
- $ pip install ray[rllib]

7. setuptools~=61.2.0
