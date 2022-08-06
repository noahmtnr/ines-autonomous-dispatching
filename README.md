# InES Team Project SS22 - Self-Organising Dispatching System

## Authors:

Agnieszka Lenart, alenart@mail.uni-mannheim.de

Maren Ehrgott, mehrgott@mail.uni-mannheim.de

Denisa Dragota, denisa.dragota@stud.ubbcluj.ro

Cosmina Ionut, cosmina.ionut@stud.ubbcluj.ro

Noah Mautner, nmautner@mail.uni-mannheim.de

Lukas Kirchdorfer, lkirchdo@mail.uni-mannheim.de

University of Mannheim, August 7th, 2022

## Introduction

### Problem Definition

In today's delivery systems, customers order at a store and the store then uses its own delivery service or an external
delivery provider (e.g., Lieferando for food delivery) to deliver the order within a certain deadline to the customer.
While these delivery services provide a fast and reliable delivery for the customers, they come with multiple issues.
They cause a high amount of commercial trips, lead to more vehicles on the roads and require parking space, which is
spares in big cities. The deliveries also increase environmental pollution due to their emissions. For small local
stores, using an own or external delivery provider might also proof to be too expensive, but not offering delivery
services could be critical to the store's success as fast online trade is becoming more and more important. In order to
overcome these problems, an alternative to traditional delivery systems is needed.

### Goals

In our project, we aim to establish a delivery system that consists of a decentralized dispatching of hitchhiking boxes
containing customer orders. The boxes are to be delivered from any store to any customer location (only given these
start and final coordinates) within a certain time period within New York City. For this, multi-hop ride sharing (i.e.,
a box rides on vehicle for some time and then transfers to another ride) using available taxi rides provided by the city
of New York is to be implemented. The system is to be trained using historical training data and tested on random and
specific custom orders. This approach is supposed to satisfy society's mobility and logistics needs (e.g., high demand
and lower costs for a ride), challenge the traditional work organization (e.g., saving money for external delivery
providers) and improve environmental protection as well as urban quality of life (e.g., less traffic).

### Methodology

As New York City is to be the setting of the hitchhike system, we constructed the street network of the city as a graph
with nodes (location consisting of latitude, longitude and a node ID) and edges (including speed limits and lengths of
edges). In this graph we strategically place hubs according to store locations, customer population distribution and the
mostly travelled nodes by the taxis. The graph provides the environment for delivering our boxes. Start and final
location of delivery are put in as coordinates, which are mapped to the nearest node, which then is mapped to the
nearest hub. At the current status, we only provide hub-to-hub delivery, which is why this mapping needs to be done.
Having the mapping of start and final position on the graph, we initialize the time with the current time and keep track
of the deadline (24 hours) to which the box has to be delivered, otherwise the delivery is conceived as failed. Having
the current position and time as input, we aim to push the box into the direction of the final hub by only taking
available shared rides. The available rides are provided from the historical taxi trip data of the city of New York,
which was pre-processed and saved into a database. The database is efficiently accessed via SQL views and queries to get
the available trips (and respective timestamps). Having access to the available trips at a certain hub at a certain
time, the box autonomously decides whether to wait, take a trip to some hub, or, in case the deadline is only 2 hours
away, book an own trip directly to the final hub. For this, the box is implemented as a Reinforcement Learning Agent (
more on that in section "Term Definitions"). It is trained on the historical trip data and its performance in this
training is measured with multiple metrics (more on that in section "Instructions for Training"). In order to finally
test the performance on new random and specific custom orders (which we generate), an agent can be compared with
benchmarks and other RL agents regarding multiple metrics and the agent's performance and routes taken are visually
displayed on a dashboard (more on that in section "Instructions for Testing").

## Foundation/Pre-Work

This section delineates basic terms and principles, as well as a tutorial for installing all required libraries and
systems. It serves to support the understanding of the following sections and the execution of the system.

### Tutorials for Required Installations

Donwload Anaconda: https://www.anaconda.com/products/individual

1. Install Python version 3.9 in anaconda prompt

- $ conda install python = 3.9

3. create new environment

- $ conda create --name ines-ad --file requirements.txt

4. activate environment

- $ conda activate ines-ad

## Repository Structure

This repository has the following folders:

**Manhattan_Graph_Environment**  – the main folder with the up-to-date version of environment and training and testing
files:

- **assets** is a folder with styling for dashboard

- **graphs** is a folder with files to create graphs used for representing the New York City

- **gym_graphenv** is a folder with RL Environment

- **testing** is a folder with files to run tests

- **training** is a folder with files to run training of agents

**archive** – contains old files that are not anymore in use

**config** – a folder with necessary configuration

**data** – a folder with data:

- **graphs** contains graphml files for representing the New York City

- **hubs**  contains files with coordiantes of hubs

- **others**  contains other data files

- **trips** contains data files with Taxi trips

**doc** - a folder with additional documentation

**hubs definition** – a folder with files that were used to determine the location of hubs

**preprocessing**  - a folder with files used to preprocess taxi trips and generate orders

**tmp** – a folder that contains checkpoints that can be used for testing the agent

## Database

For training and testing we used a MySQL database for retrieving our hubs and trip data. The MySQL database can be
accessed either through a database installed on the local machine or through a remote database in the azure cloud. To
inspect the schemas, a tool like MySQLWorkbench can help. To connect, use the following credentials:
host="mannheimprojekt.mysql.database.azure.com"
user="mannheim"
password="Projekt2022"
database="mannheimprojekt"

The database consists of Tables:

* hubs: nodeids of all hubs (used for views and retrieved in runtime)
* trips: entire trip data for all trips
* trips_routes: locations of all taxis at all times with reference to trip (used to identify share opportunities for
  package)

Views:

* filtered_trips_view: filters all on route locations from trips_routes by hubs
* filtered_trips_view_1,filtered_trips_view_2,filtered_trips_view_3, ..., :range-partitions filtered_trips_view by
  biweekly timewindows retrieved in runtime for training/testing within a two weeks window

## Environment and Reinforcement Learning

In this paragraph, we present what our RL setting looks like.

### Agent

Our agent is an intelligent box that can make decisions on its own. After arriving at a hub, it looks at the observation
space and chooses one action from the action space. It learns thanks to the RL methods by looking at the reward that it
receives after each step.

### Action Space

We modelled action space to be a one-hot-encoded vector of the length of the number of hubs. It illustrates that the
agent can move to every other hub and itself (by waiting). The action taken by the agent is always a number that is
equal to the number of the hub it is brought to.

### Observation Space

Our observation space consists of the following elements:

- ‘remaining_distance’ is a vector of the length of the number of hubs. It tells the agent how far from the destination
  hub each hub is.
- ‘final_hub’ is a one-hot-encoded vector that tells the agent which is the destination hub.
- ‘distinction’ is a binary vector of the length of action space. Each value in it shows what kind of action it is. If
  there is -1, it means that it is ‘book own ride’, 0 – ‘wait’ and 1 – ‘take a shared ride’.
- ‘allow_bookown’ is a vector of length 2 which indicates whether ‘book own rode’ is at all allowed.

Observation space is updated after each step as most information changes then.

### RL Methods

For the training of our agent, we used the following methods:

- [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf) - a policy gradient method for
  reinforcement learning.


- [Deep Q-Network (DQN )](https://arxiv.org/pdf/1312.5602v1.pdf) - approximates a state-value function in a Q-Learning
  framework with a neural network.


- [Rainbow DQN ](https://arxiv.org/pdf/1710.02298v1.pdf) - an extended DQN that combines several improvements into a
  single learner.

## Instruction for Training

There are 5 different files for training:

- custom_actions.py: instantiates environment before executing custom or random actions (no training, more for
  debugging)
- train_DQN.py: run to train agent using DQN algorithm
- train_PPO.py: run to train agent using PPO algorithm
- train_Rainbow.py: run to train agent using Rainbow algorithm
- train_Rainbow_custom_unnormalized: run to train agent using Rainbow algorithm with custom Neural Network architecture
  implemented in Tensorflow

However, the best training results seen so far were achieved by using train_Rainbow.py. Therefore, we highly recommend
using this algorithm for the Manhattan Environment. The following "Parameters" section describes how to adapt the
training file.

### Parameters

In train_Rainbow.py you can set the following hyperparameters for the Rainbow algorithm:

- "framework": by default Tensorflow, use "torch" if you want to use PyTorch
- "num_workers": set to >0 if you wan multiple-process operation
- "hiddens": define number of neurons and layers in list, e.g. [100,50] for neural network with 2 hidden layers
  consisting of 100 and 50 neurons respectively
- "fcnet_activation": define activation function, e.g. "relu"
- "train_batch_size": batch size used for training
- "gamma": discount factor of the Markov Decision Process
- "num_atoms": number of atoms for representing the distribution of return; when this is greater than 1, distributional
  Q-learning is used
- "v_min" and "v_max": expected range of reward returns

Other training hyperparameters:

- "n_iter": number of training iterations

Further settings:

- "wandb.login()": set your personal wandb key in order to see the training metrics in the wandb dashboard
- "wandb.init()": define the name of your wandb project

The checkpoints of the training are saved in /tmp/rainbow-new/rllib_checkpoint which can then be used for testing.

### Results of Training

The actions and corresponding routes can be found in log-files on WandB. To open a log-file, first select a project,
then select a run and then select "Logs" on the menu bar on the left.

Multiple WandB metrics (see https://wandb.ai/hitchhike/Comparison_Hypothese_Normalisiert) are used to measure the
training performance:

**Episodes and Steps.**

- *count_steps_mean* : Average number of steps the agent takes for one order.
- *n_trained_episodes*: Number of episodes the agent trained.

**Delivered and Not Deliver.**

- *count_terminated* : Number of orders that was terminated (interrupted because bookown was made).
- *count_delivered_on_time* : Number of orders that was delivered within the pre-specified period of delivery.
- *count_delivered_with_delay* : Number of orders that was deliverd within 12 hours after the pre-specified period of
  delivery.
- *count_not_deliverd* : Number of orders that was not delivered within the pre-specified period of delivery (plus 12
  hours).
- *share_delivered_on_time* : Ratio of orders that was delivered on time to the total number of orders.

**Available and Useful Trips.**
Available equals the shared trips that were available to an agent in one run. Available useful equals the useful shared
trips available. Useful means that taking the respective trip reduces the remaining distance to the final hub.

- *count_shared_available* : Number of steps in which any shared ride is available.
- *ratio_shared_available_to_all_steps* : Ratio of the number of steps where any kind of shared trip is available to the
  number of steps in total. Shows how often shared trips are possible. Reflects the sparseness of trips over time.
- *count_shared_available_useful* : Number of steps in which any useful shared ride is available.
- *shared_available_useful_to_shared_available* : Ratio of the number of steps where useful trips are available to the
  number of steps where any shared ride is available.
- *shared_taken_to_shared_available* : Ratio of the number of steps where a shared ride is taken to the number of steps
  where any shared ride is available.
- *ratio_shared_available_to_all_steps* : Ratio of the number of steps where a shared ride is available to the total
  number of steps.
- *shared_taken_useful_to_shared_available_useful* : Ratio of the number of steps where a useful shared ride is taken to
  the number of steps where a useful shared ride is available.

**Reward**

- *mean_reward* : Average reward an agent received for the episode.
- *max_reward* : Maximum reward an agent received for the episode.

**Bookowns, Shares and Waits.**

- *boolean_has_booked_any_own*: States whether an own taxi ride was booked for an order
- *ratio_delivered_without_bookown_to_all_delivered* : Share of total rides that was delivered without any own rides
  booked
- *share_of_bookown_mean* : Average share of own rides to all steps taken for orders
- *share_mean* : Average number of own rides taken for orders
- *share_to_own_ratio_mean* : Average ratio of share rides to own rides, i.e. how many share rides where taken per own
  ride
- *share_to_own_ratio_max* : Maximum ratio of share rides to own rides, i.e. how many share rides where taken per own
  ride
- *wait_mean* :  Average number of wait steps taken for orders
- *share_of_wait_mean* : Average share of wait steos to all steps taken for orders
- *share_of_share_mean* : Average share of shared rides to all steps taken for orders
- *own_mean* : Average number of own rides taken for orders

  Example from WandB:
  ![grafik](https://user-images.githubusercontent.com/93478758/182628279-220e1217-2c11-4bc5-ab97-57f666af62ff.png)

**Distance Reduced.**

- *distance_reduced_with_ownrides* : Distance to final hub reduced by booking own taxi rides, i.e.
  shortest-path-from-pickup-to-dropoff - distance-covered-with-shared-rides
- *bookown_distance_not_covered* : how much distance we don't have to ride with book own
- *distance_reduced_with_shared* : Distance to final hub reduced by sharing existing taxi rides, i.e.
  shortest-path-from-pickup-to-dropoff - distance-covered-with-own-rides
- *bookown_distance_not_covered_share* : how much distance (in % of total distance) we don't have to ride with book own
- *distance_reduced_with_shared_share* : Share of total distance from pickup to dropoff that was reduced by sharing
  rides

### Instructions for Testing

Testing can be done for one agent in more detail (see section "Detailled Individual Testing") or by comparing multiple
agents on multiple metrics more generelly (see section "Benchmarking and Comparison").

**Configuring Agents and Running Comparison**
We have multiple benchmarks to which a Reinforcement Learning Agent can be compared to.

- Bookown Agent: Books 1 own ride from the start to the final hub. Finished.
- Random Agent: Takes a random action (i.e., chooses any of the hubs) in each step.
- Shares Agent: Takes the shared ride that reduces the remaining distance the most in each step. If no shared ride is
  available, the agent waits at the current hub.
- SharedBookEnd Agent: Takes the shared ride that reduces the remaining distance the most in each step. If no shared
  ride is available, the agent waits at the current hub. If the agent hasn't reached the final hub two hours before
  deadline, he is forced to book an own ride to the final hub.

Note that the Cost Agent is no longer in use as the observation space changed over the course of the project.

For the Reinforcement Learning Agent, one can configure the training checkpoint from which the agent should be restored
and tested on. For this, navigate to the file of the respective agent in the testing folder. Go to the method
run_one_episode(). Adapt the file_name (i.e., file path) to your selected checkpoint.

For the benchmark agents and the RL agents, one can run a comparison which tests all agents for certain orders and
outputs rankings for multiple metrics on the performance of the agents on these orders. The following steps need to be
conducted in order to run the comparison:

1. Open the BenchmarkWrapper.py file in the testing folder.
2. Go to the method file_read(self).
3. If you want to test on random orders, then uncomment the first section of the method and comment out the second
   section. If you want to test on specific orders (which we have selected previously), then comment out the first
   section and uncomment the second section of the method. Specify the number of orders you want to test by changing the
   parameter "nrows" to the respective number of orders.
4. Open the comparison.py file in the testing folder.
5. Move to the end of the file and configure which agents you want to compare by removing the hashtags for comments or
   commenting out certain agents. Then select a comparer. Currently the comparer for 5 agents (random, rainbow, shares,
   bookown, sharesbookend) is enabled.
6. Run the comparison.py file. Depending on the number of agents and the number of orders to be tested this might take a
   while (approx. 3 hours for 5 agents and 11 orders).

**Results of Comparison**
Finally, the comparison outputs an overview of all agents, as well as multiple rankings on performance metrics. These
metrics are:

- Number of hubs travelled.
- Distance travelled.
- Number of Book Owns.
- Ratio of Shares to Book Owns.
- Reward.
- Travel Time.
- Ratio Distance Reduced with Shares to Whole Distance. The rankings always show the best agent on rank 1.

**Detailled Individual Testing**
The dashboard can be used to visually understand the actions that our trained agent has taken in test orders. To open
the dashboard locally in your browser, you just need to run the file double_trouble.py. The dashboard consists of two
tabs: static and interactive. The static visualization shows the route that the agent has taken on the map of New York,
as well as some order statistics such as number of actions taken. In the interactive tab, the user can manually perform
actions and compare them to the actions taken by the agent on different test cases. The following GIF demonstrates how
to operate in the interactive tab:

![](https://github.com/noahmtnr/ines-autonomous-dispatching/blob/comments/Dashboard.gif)



