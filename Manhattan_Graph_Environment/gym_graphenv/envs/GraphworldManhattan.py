import pickle
import random
import sys
import time
from datetime import datetime, timedelta

import gym
import modin.pandas as pd
import numpy as np
import osmnx as ox
# from mpl_toolkits.basemap import Basemap
import plotly.express as px
from gym import spaces

RAY_ENABLE_MAC_LARGE_OBJECT_STORE = 1.
sys.path.insert(0, "")
# from config.definitions import ROOT_DIR
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
from typing import Dict

from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy

from Manhattan_Graph_Environment.graphs.ManhattanGraph import ManhattanGraph
from Manhattan_Graph_Environment.graphs.LearnGraph import LearnGraph
from Manhattan_Graph_Environment.database_connection import DBConnection


class GraphEnv(gym.Env):
    REWARD_AWAY = -1
    REWARD_GOAL = 100
    WAIT_TIME_SECONDS = 300
    WAIT_TIME_MINUTES = WAIT_TIME_SECONDS / 60

    def __init__(self, use_config: bool = True):
        # Range partitioning boundaries for retrieval of taxi rides from db. Don't change until more data is stored in db.
        DB_LOWER_BOUNDARY = '2016-01-01 00:00:00'
        DB_UPPER_BOUNDARY = '2016-01-14 23:59:59'
        self.LEARNGRAPH_FIRST_INIT_DONE = False

        self.done = False

        if (use_config):
            self.env_config = self.read_config()
        else:
            self.env_config = None

        self.distance_matrix = None

        self.DB = DBConnection()
        hubs = self.DB.fetch_all_hubs()
        manhattan_graph = ManhattanGraph(filename='simple', hubs=hubs)
        self.manhattan_graph = manhattan_graph

        self.hubs = manhattan_graph.hubs
        self.n_hubs = len(hubs)

        self.trips = self.DB.fetch_all_available_trips(DB_LOWER_BOUNDARY, DB_UPPER_BOUNDARY)
        print(f"Initialized with {len(self.hubs)} hubs and {len(self.trips)} taxi rides within two weeks")

        self.state = None
        self.state_of_delivery = DeliveryState.IN_DELIVERY
        self.allow_bookown = 0
        self.route_travel_distance = 0

        self.action_space = gym.spaces.Discrete(self.n_hubs)

        self.observation_space = spaces.Dict(
            {
                'remaining_distance': gym.spaces.Box(low=np.zeros(self.n_hubs) - 200000,
                                                     high=np.zeros(self.n_hubs) + 200000, shape=(self.n_hubs,),
                                                     dtype=np.float64),
                'final_hub': gym.spaces.Box(low=0, high=1, shape=(self.n_hubs,), dtype=np.float64),
                'distinction': gym.spaces.Box(low=np.zeros(self.n_hubs) - 1, high=np.zeros(self.n_hubs) + 1,
                                              shape=(self.n_hubs,), dtype=np.float64),
                'allow_bookown': gym.spaces.Discrete(2)
            })

        self.rem_distance_values = []
        self.rd_mean = 120.2
        self.rd_stdev = 5571.48

        self.distance_covered_with_shared = 0
        self.distance_covered_with_ownrides = 0

    def get_Graph(self):
        return self.manhattan_graph

    def one_hot(self, pos: int):
        """Returns a one hot vector with 1 at position pos.

        Args:
            pos (int): Number represented by the one hot vector

        Returns:
            array([ 0.,  0.,  1.,  0.,  0.]): Vector of floats 
        """
        one_hot_vector = np.zeros(len(self.hubs))
        one_hot_vector[pos] = 1
        return one_hot_vector

    def reset(self, start_parameters: {} = None):
        """
        Reset function used to reset the environment. 
        If start_parameters is given, resets to the passed order. Else if self.env_config is given, resets to the order given in the env_config. Else initializes order with random pickup, dropoff and time
        Has to be called once after initializing environment to initialize state. Called by ray rllib agent once after setup and at the end of every episode.

        Args:
            start_parameters (None, optional): Optional dictionary of order parameters to reset the environment to. Dictionary has to consist of: .
                {
                    pickup: int (hubId of the pickup hub, example: 25),
                    dropoff: int (hubId of the dropoff hub, example: 70),
                    time = datetime (example: datetime.strptime('2016-01-03 14:00:00', '%Y-%m-%d %H:%M:%S'))
                    deadline = datetime (example: datetime.strptime('2016-01-03 17:00:00', '%Y-%m-%d %H:%M:%S'))
                }
        Returns:
            self.state
        """

        resetExecutionStart = time.time()
        self.route_taken = []

        if (start_parameters != None):
            print("Started Reset() with start parameters")
            self.final_hub = start_parameters['dropoff']
            self.start_hub = start_parameters['pickup']
            self.position = self.start_hub
            self.pickup_time = start_parameters['time']
            self.time = self.pickup_time
            self.deadline = start_parameters['deadline']
            self.total_travel_time = 0
        else:
            if (self.env_config == None or self.env_config == {}):
                print("Started Reset() with random order")
                self.final_hub = random.randint(0, self.n_hubs - 1)
                self.start_hub = random.randint(0, self.n_hubs - 1)

                if (self.start_hub == self.final_hub):
                    self.start_hub = random.randint(0, self.n_hubs - 1)  # just in case ;)

                self.position = self.start_hub

                pickup_day = np.random.randint(low=1, high=14)
                pickup_hour = np.random.randint(24)
                pickup_minute = np.random.randint(60)
                self.pickup_time = datetime(2016, 1, pickup_day, pickup_hour, pickup_minute, 0)
                self.time = self.pickup_time
                self.total_travel_time = 0
                self.deadline = self.pickup_time + timedelta(hours=24)
                self.current_wait = 1  ## to avoid dividing by 0
            else:
                self.env_config = self.read_config()
                print("Started Reset() with config")
                print(self.env_config)
                self.final_hub = self.env_config['delivery_hub_index']

                self.start_hub = self.env_config['pickup_hub_index']

                self.position = self.start_hub

                self.pickup_time = self.env_config['pickup_timestamp']
                self.time = datetime.strptime(self.pickup_time, '%Y-%m-%d %H:%M:%S')
                self.total_travel_time = 0
                self.deadline = datetime.strptime(self.env_config['delivery_timestamp'], '%Y-%m-%d %H:%M:%S')
                self.current_wait = 0
        self.distance_covered_with_shared = 0
        self.distance_covered_with_ownrides = 0
        self.distance_reduced_with_shared = 0  # to final hub
        self.distance_reduced_with_ownrides = 0  # to final hub

        print(f"Reset pickup: {self.position} dropoff: {self.final_hub} time: {self.time}")

        learn_graph = LearnGraph(n_hubs=self.n_hubs, manhattan_graph=self.manhattan_graph, final_hub=self.final_hub)
        self.learn_graph = learn_graph

        if (self.LEARNGRAPH_FIRST_INIT_DONE == False):
            self.distance_matrix = self.learn_graph.fill_distance_matrix()

        self.LEARNGRAPH_FIRST_INIT_DONE = True
        self.learn_graph.add_travel_cost_and_distinction_layer(self.availableTrips(), self.distance_matrix)
        self.learn_graph.add_remaining_distance_layer(current_hub=self.position, distance_matrix=self.distance_matrix)

        self.count_hubs = 0
        self.count_actions = 0
        self.count_wait = 0
        self.count_bookown = 0
        self.booked_own = 0

        self.count_share = 0
        self.count_steps = 0
        self.action_choice = None

        self.old_position = self.start_hub
        self.current_trip = None

        self.own_ride = False
        self.has_waited = False
        reward = 0

        if ((self.deadline - self.time).total_seconds() / 60 <= 120):
            self.allow_bookown = 1
        else:
            self.allow_bookown = 0

        # new metrics for shares and bookowns
        self.count_shared_available = 0
        self.boolean_shared_available = 0
        self.count_shared_available_useful = 0
        self.count_shared_taken_useful = 0
        self.boolean_useful_shares_available = 0
        self.orders_delivered_without_booked = 0

        self.state = {
            'remaining_distance': ((self.learn_graph.adjacency_matrix('remaining_distance')[
                                        self.position] - self.rd_mean) / self.rd_stdev).astype(np.float64),
            'final_hub': self.one_hot(self.final_hub).astype(np.float64),
            'distinction': self.learn_graph.adjacency_matrix('distinction')[self.position].astype(np.float64),
            'allow_bookown': self.allow_bookown,
        }
        self.rem_distance_values.extend(
            self.learn_graph.adjacency_matrix('remaining_distance')[self.position].astype(np.float64))

        self.shortest_distance = self.learn_graph.adjacency_matrix('remaining_distance')[self.position][self.final_hub]

        resetExecutionTime = (time.time() - resetExecutionStart)
        # print(f"Reset() Execution Time: {str(resetExecutionTime)}")
        return self.state

    def step(self, action: int):
        """ Executes an action based on the index passed as a parameter (only works with moves to direct neighbors as of now)
        Args:
            action (int): index of action to be taken from availableActions
        Returns:
            int: new position
            int: new reward
            boolean: isDone
        """

        startTime = time.time()

        # if agent is in time window 2 hours before deadline, we just move him to the final hub
        if ((self.deadline - self.time).total_seconds() / 60 <= 120):
            self.allow_bookown = 1
            action = self.final_hub
            print("Force Manual Delivery")
        else:
            self.allow_bookown = 0

        # determine whether shared ride was useful (= whether remaining distance was reduced)
        # compute the number of shared available actions and the number of useful shared available actions
        counter = 0
        boolean_available_temp = False
        boolean_useful_temp = False
        for hub in self.shared_rides_mask:
            if hub == 1:
                # count on step-base
                boolean_available_temp = True
                self.count_shared_available += 1
                # check whether remaining distance decreases with new position
                if self.state["remaining_distance"][counter] > 0:
                    # count on ride-base
                    self.count_shared_available_useful += 1
                    # count on step-base
                    boolean_useful_temp = True

        # set old position to current position before changing current position
        self.old_position = self.position
        step_duration = 0

        if self.validateAction(action):
            startTimeWait = time.time()
            self.count_steps += 1
            if (action == self.position):
                # action = wait
                step_duration = self.WAIT_TIME_SECONDS
                self.has_waited = True
                self.own_ride = False
                self.count_wait += 1
                self.action_choice = "Wait"
                executionTimeWait = (time.time() - startTimeWait)
                route_taken = [self.manhattan_graph.get_nodeid_by_hub_index(self.position)]
                pass

            else:
                pickup_nodeid = self.manhattan_graph.get_nodeid_by_hub_index(self.position)
                dropoff_nodeid = self.manhattan_graph.get_nodeid_by_hub_index(action)
                route = ox.shortest_path(self.manhattan_graph.inner_graph, pickup_nodeid, dropoff_nodeid,
                                         weight='travel_time')
                route_taken = [pickup_nodeid]
                route_taken.extend(route)
                route_taken.append(dropoff_nodeid)
                route_travel_time = ox.utils_graph.get_route_edge_attributes(self.manhattan_graph.inner_graph, route,
                                                                             attribute='travel_time')
                self.route_travel_distance = sum(
                    ox.utils_graph.get_route_edge_attributes(self.manhattan_graph.inner_graph, route,
                                                             attribute='length'))

                if (self.learn_graph.wait_till_departure_times[(self.position, action)] == self.WAIT_TIME_SECONDS):
                    step_duration = sum(
                        route_travel_time) + self.WAIT_TIME_SECONDS  # we add 5 minutes (300 seconds) so the taxi can arrive
                elif (self.learn_graph.wait_till_departure_times[(self.position, action)] != self.WAIT_TIME_SECONDS and
                      self.learn_graph.wait_till_departure_times[(self.position, action)] != 0):
                    step_duration = sum(route_travel_time)
                    departure_time = self.learn_graph.wait_till_departure_times[(self.position, action)]
                    self.current_wait = (departure_time - self.time).seconds
                    self.time = departure_time
                if (self.shared_rides_mask[action] == 1):
                    self.count_share += 1
                    self.action_choice = "Share"

                    # check whether current action is useful
                    if self.state["remaining_distance"][action] > 0:
                        self.count_shared_taken_useful += 1
                    self.distance_covered_with_shared += self.route_travel_distance
                    self.distance_reduced_with_shared += \
                    self.learn_graph.adjacency_matrix('remaining_distance')[self.old_position][action]


                else:
                    self.count_bookown += 1
                    self.action_choice = "Book"
                    self.distance_covered_with_ownrides += self.route_travel_distance
                    self.distance_reduced_with_ownrides += \
                    self.learn_graph.adjacency_matrix('remaining_distance')[self.old_position][action]

                startTimeRide = time.time()
                self.has_waited = False
                self.count_hubs += 1

                self.old_position = self.position
                self.position = action

                executionTimeRide = (time.time() - startTimeRide)
                # print(f"Time Ride: {str(executionTimeRide)}")
                pass

        else:
            print("invalid action")
            print("action: ", action)
            print("action space: ", self.action_space)

        # refresh travel cost layer after each step
        self.learn_graph.add_travel_cost_and_distinction_layer(self.availableTrips(), self.distance_matrix)
        self.learn_graph.add_remaining_distance_layer(current_hub=self.position, distance_matrix=self.distance_matrix)

        startTimeLearn = time.time()
        self.old_state = self.state
        self.state = {
            'remaining_distance': (((self.learn_graph.adjacency_matrix('remaining_distance')[
                self.position]) - self.rd_mean) / self.rd_stdev).astype(np.float64),
            'final_hub': self.one_hot(self.final_hub).astype(np.float64),
            'distinction': self.learn_graph.adjacency_matrix('distinction')[self.position].astype(np.float64),
            'allow_bookown': self.allow_bookown,
        }

        self.time += timedelta(seconds=step_duration)
        self.count_actions += 1

        reward, self.done, self.state_of_delivery = self.compute_reward(action)

        executionTime = (time.time() - startTime)

        if self.count_bookown > 0:
            self.booked_own = 1

        # counting on step-base (not individual ride-base)
        if boolean_available_temp == True:
            self.boolean_shared_available += 1
        if boolean_useful_temp == True:
            self.boolean_useful_shares_available += 1

        return self.state, reward, self.done, {"timestamp": self.time, "step_travel_time": step_duration,
                                               "distance": self.distance_matrix[self.old_position][self.position],
                                               "count_hubs": self.count_hubs, "action": self.action_choice,
                                               "hub_index": action, "route": route_taken, "remaining_dist":
                                                   self.learn_graph.adjacency_matrix('remaining_distance')[
                                                       self.position][self.final_hub],
                                               "dist_covered_shares": self.distance_covered_with_shared,
                                               "dist_covered_bookown": self.distance_covered_with_ownrides}

    def compute_reward(self, action: int):
        """ Computes the reward for an action given the state.

        Args:
            action (int): _description_

        Returns:
            _type_: _description_
        """

        distance_gained = self.old_state['remaining_distance'][self.position]
        old_distinction = self.old_state['distinction']
        cost_of_action = self.learn_graph.adjacency_matrix('cost')[self.old_position][action]
        self.done = False

        action_type = None
        bookown = False
        wait = False
        share = False
        reward = 0

        if (old_distinction[action] == -1):  # book own
            action_type = ActionType.OWN
        elif (old_distinction[action] == 0):  # wait
            action_type = ActionType.WAIT
        else:
            action_type = ActionType.SHARE

        if (self.position == self.final_hub):  # arrived at final hub
            self.done = True
            if ((self.deadline - self.time).total_seconds() / 60 >= 120):  # on time
                state_of_delivery = DeliveryState.DELIVERED_ON_TIME
                print(
                    f"DELIVERED IN TIME AFTER {self.count_actions} ACTIONS (#wait: {self.count_wait}, #share: {self.count_share}, #book own: {self.count_bookown})")
                if (action_type == ActionType.OWN):
                    if (self.allow_bookown == 0):
                        reward = -500
                    else:
                        reward = 1000
                elif (action_type == ActionType.SHARE):
                    # high reward if agent comes to final hub with shared ride
                    reward = 10000

            else:  # with delay (does not occur with backup book own)
                state_of_delivery = DeliveryState.DELIVERED_ON_TIME
                print(
                    f"MANUAL DELIVERY WITH {(self.deadline - self.time).total_seconds() / 60} MINUTES TO DEADLINE - ACTIONS (#wait: {self.count_wait}, #share: {self.count_share}, #book own: {self.count_bookown})")
                reward = 1000

        else:  # not yet arrived at final hub
            self.done = False
            state_of_delivery = DeliveryState.IN_DELIVERY
            if (action_type == ActionType.WAIT):
                reward = 100
            elif (action_type == ActionType.OWN):
                if (self.allow_bookown == 0):
                    self.done = True
                    state_of_delivery = DeliveryState.TERMINATED
                    reward = -1000
                    print("TERMINATED Because of BOOKED OWN NOT TO FINAL")
                else:
                    # dieser Case kann eigentlich nicht eintreten
                    reward = (distance_gained / 100) * 100
            elif (action_type == ActionType.SHARE):
                reward = distance_gained * 100 + 100

        print(self.old_position, "->", action, "SHARE" if action_type == 1 else "BOOK" if action_type == 2 else "WAIT",
              "Distance:", distance_gained, "Reward:", reward)

        return reward, self.done, state_of_delivery

    def availableTrips(self, time_window: int = 5, oversample_by_n_rides: int = 0):
        """Returns a list of all available trips at the current node and within the next time_window minutes. For each shared ride available, includes the time of departure from the current node, the target node of the ride and the route to the target node.

        Args:
            time_window (int, optional): Time window in which to look for available rides. Defaults to 5.
            oversample_by_n_rides (int, optional): Number of synthetically generated rides that should be added to available rides from db. Defaults to 0.

        Returns:
            [{'departure_time': datetime, 'target_hub': int, 'route': [int]}]: List of available shared rides
        """
        startTime = time.time()
        list_trips = []
        position = self.manhattan_graph.get_nodeid_by_hub_index(self.position)
        position_str = str(position)
        final_hub_postion = self.manhattan_graph.get_nodeid_by_index(self.final_hub)

        start_timestamp = self.time
        end_timestamp = self.time + timedelta(minutes=time_window)

        trips = self.trips

        for tripId, nodeId, timestamp in trips:
            if (nodeId == position):
                if timestamp <= end_timestamp and timestamp >= start_timestamp:
                    route, times = self.DB.fetch_route_from_trip(tripId)
                    isNotFinalNode = True
                    if isNotFinalNode:
                        index_in_route = route.index(position)
                        position_timestamp = datetime.strptime(times[index_in_route], '%Y-%m-%d %H:%M:%S')
                        route_to_target_node = route[index_in_route::]
                        hubsOnRoute = any(node in route_to_target_node for node in self.hubs)
                        if hubsOnRoute:
                            route_hubs = [node for node in route_to_target_node if node in self.hubs]
                            for hub in route_hubs:
                                index_hub_in_route = route.index(hub)
                                index_hub_in_route += 1
                                route_to_target_hub = route[index_in_route:index_hub_in_route]
                                if (hub != position):
                                    trip = {'departure_time': position_timestamp, 'target_hub': hub,
                                            'route': route_to_target_hub, 'trip_row_id': tripId}
                                    list_trips.append(trip)
        self.available_actions = list_trips

        shared_rides_mask = np.zeros(self.n_hubs)
        for i in range(len(list_trips)):
            shared_rides_mask[self.manhattan_graph.get_hub_index_by_nodeid(list_trips[i]['target_hub'])] = 1

        self.shared_rides_mask = shared_rides_mask
        if (oversample_by_n_rides != 0):
            self.available_actions += self.generate_rides(
                oversample_by_n_rides)  # Uses shared_rides_mask to oversample available shares and writes them into shared_rides_mask and available_actions

        executionTime = (time.time() - startTime)
        return list_trips

    def generate_rides(self, n_rides: int = 40):
        """Function used in training to increase the number of available shared rides available at each step. Returns n_rides synthetic rides in a list of the same format as availableTrips() returns.

        Args:
            n_rides (int, optional): Number of rides that should be generated. Defaults to 40.

        Returns:
            [{'departure_time': datetime, 'target_hub': int, 'route': [int]}]: List of generated rides
        """
        list_trips = []
        depart_time_list = []
        target_hub_list = []
        route_list = []

        # hubs where shared rides go to
        shared_ride_hubs = []
        for i in range(len(self.shared_rides_mask)):
            if (self.shared_rides_mask[i] == 1):
                shared_ride_hubs.append(i)
        # create list of hubs where synthetic rides should go to
        sampled_hub_list = np.random.choice(self.n_hubs, 40, replace=False)
        for i in range(len(sampled_hub_list)):
            # create only shared rides to this hubs which are not already covered by a shared ride
            if (sampled_hub_list[i] not in shared_ride_hubs):
                target_hub_list.append(sampled_hub_list[i])

        for i in range(len(target_hub_list)):
            depart_time = self.time
            depart_time += timedelta(seconds=180)  # add 3 minutes wait time until departure of synthetic ride
            depart_time_list.append(depart_time)

            route = ox.shortest_path(self.manhattan_graph.inner_graph,
                                     self.manhattan_graph.get_nodeid_by_hub_index(self.position),
                                     self.manhattan_graph.get_nodeid_by_hub_index(target_hub_list[i]),
                                     weight='travel_time')
            route_list.append(route)

        generated_trips_list = []
        for i in range(len(target_hub_list)):
            trip = {'departure_time': depart_time_list[i], 'target_hub': target_hub_list[i], 'route': route_list[i]}
            generated_trips_list.append(trip)

        for hub in target_hub_list:
            self.shared_rides_mask[hub] = 1

        return generated_trips_list

    def validateAction(self, action: int):
        """Validates if given action is valid 
        Args:
            action (int): final hub to move to

        Returns:
            boolean: is action valid or not
        """
        return action < self.n_hubs

    def read_config(self):
        """Reads config file that includes preset order parameters (pickup,dropoff and time) and writes it into self.env_config

        Returns:
            dict: retrieved config dictionary
        """
        filepath = "env_config.pkl"
        with open(filepath, 'rb') as f:
            loaded_dict = pickle.load(f)
        self.env_config = loaded_dict
        return loaded_dict

    def render(self):
        """Renders the current state of the environment in a plotly figure.
        """

        shared_rides = list()
        shared_ids = list()

        for i, trip in enumerate(self.availableTrips()):
            shared_ids.append(trip['target_hub'])

        all_hubs = self.hubs
        book_own_ids = list(set(all_hubs) - set(shared_ids))

        position = self.manhattan_graph.get_nodeid_by_index(self.position)
        final = self.manhattan_graph.get_nodeid_by_index(self.final_hub)
        start = self.manhattan_graph.get_nodeid_by_index(self.start_hub)

        node_sizes = list()
        actions = []
        for n in self.manhattan_graph.nodes():
            if n == position:
                actions.append('position')
                node_sizes.append(120)
            else:
                if n == final:
                    actions.append('final')
                    node_sizes.append(120)
                else:
                    if n == start:
                        actions.append('start')
                        node_sizes.append(120)
                    else:
                        if n in shared_ids:
                            actions.append('shared')
                            node_sizes.append(120)
                        else:
                            if n in book_own_ids:
                                actions.append('book')
                                node_sizes.append(120)
                            else:
                                # colors.append('w')
                                node_sizes.append(0)
        df = pd.read_csv("ines-autonomous-dispatching/data/hubs/longlist.csv")
        df['actions'] = actions

        graph = self.manhattan_graph.inner_graph

        px.set_mapbox_access_token(open("mapbox_token").read())
        fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", hover_name="id", color="actions",
                                # size="car_hours",
                                color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10)
        fig.show()


class ActionType:
    WAIT, SHARE, OWN = range(3)


class DeliveryState:
    DELIVERED_ON_TIME, DELIVERED_WITH_DELAY, NOT_DELIVERED, IN_DELIVERY, TERMINATED = range(5)


class CustomCallbacks(DefaultCallbacks):
    last_count_delivered_on_time = 0
    last_count_delivered_with_delay = 0
    last_count_not_delivered = 0
    last_count_terminated = 0
    count_not_delivered = 0
    count_terminated = 0
    count_delivered_with_delay = 0
    count_delivered_on_time = 0

    # new metrics for shares and bookowns
    count_shared_available = 0
    count_shared_available_useful = 0
    count_shared_taken = 0
    last_count_bookowns = 0
    count_bookowns = 0
    count_shared_taken_useful = 0
    boolean_useful_shares_available = 0
    boolean_shared_available = 0
    orders_delivered_without_booked = 0

    def on_algorithm_init(
            self,
            *,
            algorithm,
            **kwargs,
    ):
        """Callback run when a new trainer instance has finished setup.
        This method gets called at the end of Trainer.setup() after all
        the initialization is done, and before actually training starts.
        Args:
            trainer: Reference to the trainer instance.
            kwargs: Forward compatibility placeholder.
        """
        self.count_delivered_on_time = 0
        self.count_delivered_with_delay = 0
        self.count_not_delivered = 0
        self.count_terminated = 0

        # metrics for shares and bookowns
        self.count_shared_available = 0
        self.boolean_shared_available = 0
        self.count_shared_taken = 0
        self.count_bookowns = 0

        self.count_shared_available_useful = 0
        self.count_shared_taken_useful = 0
        self.boolean_useful_shares_available = 0
        self.orders_delivered_without_booked = 0

    def on_episode_start(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: Episode,
            env_index: int,
            **kwargs
    ):
        # Make sure this episode has just been started (only initial obs
        # logged so far).
        assert episode.length == 0, (
            "ERROR: `on_episode_start()` callback should be called right "
            "after env reset!"
        )
        episode.env = base_env.get_sub_environments(as_dict=True)[0]
        episode.custom_metrics["count_not_delivered"] = 0
        episode.custom_metrics["count_delivered_with_delay"] = 0
        episode.custom_metrics["count_delivered_on_time"] = 0
        episode.custom_metrics["count_terminated"] = 0

        # metrics for shares and bookowns
        # episode.custom_metrics["count_shared_available"] = 0
        # episode.custom_metrics["count_shared_taken"] = 0
        episode.custom_metrics["boolean_has_booked_any_own"] = 0
        episode.custom_metrics["count_shared_available_useful"] = 0
        # episode.custom_metrics["count_shared_taken_useful"] = 0

    def on_episode_step(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: Episode,
            env_index: int,
            **kwargs
    ):
        # Make sure this episode is ongoing.
        assert episode.length > 0, (
            "ERROR: `on_episode_step()` callback should not be called right "
            "after env reset!"
        )

    def on_episode_end(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: Episode,
            env_index: int,
            **kwargs
    ):
        # Check if there are multiple episodes in a batch, i.e.
        # "batch_mode": "truncate_episodes".
        if worker.policy_config["batch_mode"] == "truncate_episodes":
            # Make sure this episode is really done.
            assert episode.batch_builder.policy_collectors["default_policy"].batches[
                -1
            ]["dones"][-1], (
                "ERROR: `on_episode_end()` should only be called "
                "after episode is done!"
            )

        episode.custom_metrics["count_wait"] = episode.env.count_wait
        episode.custom_metrics["count_bookown"] = episode.env.count_bookown
        episode.custom_metrics["count_share"] = episode.env.count_share
        episode.custom_metrics["count_steps"] = episode.env.count_steps
        episode.custom_metrics["share_wait"] = float(episode.env.count_wait / episode.env.count_actions)
        episode.custom_metrics["share_bookown"] = float(episode.env.count_bookown / episode.env.count_actions)
        episode.custom_metrics["share_share"] = float(episode.env.count_share / episode.env.count_actions)
        episode.custom_metrics[
            "share_to_own_ratio"] = episode.env.count_share if episode.env.count_bookown == 0 else float(
            episode.env.count_share / episode.env.count_bookown)
        episode.custom_metrics[
            "share_to_own_ratio"] = episode.env.count_share if episode.env.count_bookown == 0 else float(
            episode.env.count_share / episode.env.count_bookown)

        # metrics for shares and bookowns
        # ratio of shared taken when a shared is available
        if episode.env.boolean_shared_available == 0:
            episode.custom_metrics["shared_taken_to_shared_available"] = 0
        else:
            episode.custom_metrics["shared_taken_to_shared_available"] = float(
                episode.env.count_share / episode.env.boolean_shared_available)
        # counting the shared availables (if one is available in a step, then +1)
        episode.custom_metrics["count_shared_available"] = episode.env.boolean_shared_available
        episode.custom_metrics[
            "ratio_shared_available_to_all_steps"] = episode.env.boolean_shared_available / episode.env.count_steps
        # ratio: useful available shares (reducing remaining distance) of available shares
        if episode.env.count_shared_available == 0:
            episode.custom_metrics["shared_available_useful_to_shared_available"] = 0
        else:
            episode.custom_metrics["shared_available_useful_to_shared_available"] = float(
                episode.env.count_shared_available_useful / episode.env.count_shared_available)

        # counting the useful available shared rides
        episode.custom_metrics["count_shared_available_useful"] = episode.env.boolean_useful_shares_available
        # ratio: useful shares taken of useful shares available
        if episode.env.boolean_useful_shares_available == 0:
            episode.custom_metrics["shared_taken_useful_to_shared_available_useful"] = 0
        else:
            episode.custom_metrics["shared_taken_useful_to_shared_available_useful"] = float(
                episode.env.count_shared_taken_useful / episode.env.boolean_useful_shares_available)

        # displays 1 if any trip was booked and 0 if none was booked
        if episode.env.count_bookown > 0:
            episode.custom_metrics["boolean_has_booked_any_own"] = 1
        else:
            episode.custom_metrics["boolean_has_booked_any_own"] = 0

        if (episode.env.state_of_delivery == DeliveryState.DELIVERED_ON_TIME):
            self.count_delivered_on_time += 1
            episode.custom_metrics["count_delivered_on_time"] = self.count_delivered_on_time
            self.orders_delivered_without_booked += 1
        elif (episode.env.state_of_delivery == DeliveryState.DELIVERED_WITH_DELAY):
            self.count_delivered_with_delay += 1
            episode.custom_metrics["count_delivered_with_delay"] = self.count_delivered_with_delay
        elif (episode.env.state_of_delivery == DeliveryState.NOT_DELIVERED):
            self.count_not_delivered += 1
            episode.custom_metrics["count_not_delivered"] = self.count_not_delivered
        elif (episode.env.state_of_delivery == DeliveryState.TERMINATED):
            self.count_terminated += 1
            episode.custom_metrics["count_terminated"] = self.count_terminated

        if (self.count_delivered_on_time == 0):
            episode.custom_metrics["ratio_delivered_without_bookown_to_all_delivered"] = 0
        else:
            episode.custom_metrics["ratio_delivered_without_bookown_to_all_delivered"] = float(
                episode.env.orders_delivered_without_booked / self.count_delivered_on_time)

        # zum Vergleich ohne Abzug später
        # episode.custom_metrics["count_not_delivered_first"] = self.count_not_delivered

        # how much distance (in % of total distance) we don't have to ride with book own
        episode.custom_metrics['bookown_distance_not_covered_share'] = 1 - (
                    episode.env.distance_covered_with_ownrides / episode.env.shortest_distance)
        # how much distance we don't have to ride with book own
        episode.custom_metrics[
            'bookown_distance_not_covered'] = episode.env.shortest_distance - episode.env.distance_covered_with_ownrides

        episode.custom_metrics['distance_reduced_with_ownrides'] = episode.env.distance_reduced_with_ownrides
        episode.custom_metrics['distance_reduced_with_shared'] = episode.env.distance_reduced_with_shared

        episode.custom_metrics[
            'distance_reduced_with_ownrides_share'] = episode.env.distance_reduced_with_ownrides / episode.env.shortest_distance
        episode.custom_metrics[
            'distance_reduced_with_shared_share'] = episode.env.distance_reduced_with_shared / episode.env.shortest_distance

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        print(
            "trainer.train() result: {} -> {} episodes".format(
                trainer, result["episodes_this_iter"]
            )
        )
        # you can mutate the result dict to add new fields to return
        result["count_wait_min"] = result['custom_metrics']['count_wait_min']
        result["count_wait_max"] = result['custom_metrics']['count_wait_max']
        result["count_wait_mean"] = result['custom_metrics']['count_wait_mean']
        result["waiting_time_mean"] = result['custom_metrics']['count_wait_mean'] * 5

        result["count_bookown_min"] = result['custom_metrics']['count_bookown_min']
        result["count_bookown_max"] = result['custom_metrics']['count_bookown_max']
        result["count_bookown_mean"] = result['custom_metrics']['count_bookown_mean']
        result["count_share_min"] = result['custom_metrics']['count_share_min']
        result["count_share_max"] = result['custom_metrics']['count_share_max']
        result["count_share_mean"] = result['custom_metrics']['count_share_mean']
        result["count_steps_min"] = result['custom_metrics']['count_steps_min']
        result["count_steps_max"] = result['custom_metrics']['count_steps_max']
        result["count_steps_mean"] = result['custom_metrics']['count_steps_mean']

        result["share_wait_min"] = result['custom_metrics']['share_wait_min']
        result["share_wait_max"] = result['custom_metrics']['share_wait_max']
        result["share_wait_mean"] = result['custom_metrics']['share_wait_mean']
        result["share_bookown_min"] = result['custom_metrics']['share_bookown_min']
        result["share_bookown_max"] = result['custom_metrics']['share_bookown_max']
        result["share_bookown_mean"] = result['custom_metrics']['share_bookown_mean']
        result["share_share_min"] = result['custom_metrics']['share_share_min']
        result["share_share_max"] = result['custom_metrics']['share_share_max']
        result["share_share_mean"] = result['custom_metrics']['share_share_mean']

        result["share_to_own_ratio_min"] = result['custom_metrics']['share_to_own_ratio_min']
        result["share_to_own_ratio_max"] = result['custom_metrics']['share_to_own_ratio_max']
        result["share_to_own_ratio_mean"] = result['custom_metrics']['share_to_own_ratio_mean']

        result["count_delivered_on_time"] = result['custom_metrics'][
                                                "count_delivered_on_time_max"] - CustomCallbacks.last_count_delivered_on_time
        result["count_delivered_with_delay"] = result['custom_metrics'][
                                                   "count_delivered_with_delay_max"] - CustomCallbacks.last_count_delivered_with_delay
        if result["count_delivered_with_delay"] < 0:
            result["count_delivered_with_delay"] = 0
        """
        print("COUNTER AUSGABE")
        print("Erg:", result['custom_metrics']["count_not_delivered_max"])
        print("Abzug", self.last_count_not_delivered)
        """

        result["count_terminated"] = result['custom_metrics'][
                                         "count_terminated_max"] - CustomCallbacks.last_count_terminated
        result["count_not_delivered"] = result['custom_metrics'][
                                            "count_not_delivered_max"] - CustomCallbacks.last_count_not_delivered
        # zum Vergleich ohne Abzug
        # result["count_not_delivered_first"] = result['custom_metrics']["count_not_delivered_max"]

        CustomCallbacks.last_count_not_delivered = CustomCallbacks.last_count_not_delivered + result[
            "count_not_delivered"]
        CustomCallbacks.last_count_delivered_with_delay = CustomCallbacks.last_count_delivered_with_delay + result[
            "count_delivered_with_delay"]
        CustomCallbacks.last_count_delivered_on_time = CustomCallbacks.last_count_delivered_on_time + result[
            "count_delivered_on_time"]
        CustomCallbacks.last_count_terminated = CustomCallbacks.last_count_terminated + result["count_terminated"]

        # metrics für shares and bookowns
        result["boolean_has_booked_any_own"] = result['custom_metrics'][
            "boolean_has_booked_any_own_mean"]  # - CustomCallbacks.last_count_bookowns
        # CustomCallbacks.last_count_bookowns = CustomCallbacks.last_count_bookowns + result["count_booked_own"]
        result["shared_taken_to_shared_available"] = result['custom_metrics']["shared_taken_to_shared_available_mean"]
        result["count_shared_available"] = result['custom_metrics']["count_shared_available_mean"]
        result["ratio_shared_available_to_all_steps"] = result['custom_metrics'][
            "ratio_shared_available_to_all_steps_mean"]
        result["shared_available_useful_to_shared_available"] = result['custom_metrics'][
            "shared_available_useful_to_shared_available_mean"]
        result["shared_taken_useful_to_shared_available_useful"] = result['custom_metrics'][
            "shared_taken_useful_to_shared_available_useful_mean"]
        result["count_shared_available_useful"] = result['custom_metrics']["count_shared_available_useful_mean"]

        result["ratio_delivered_without_bookown_to_all_delivered"] = result['custom_metrics'][
            "ratio_delivered_without_bookown_to_all_delivered_mean"]

        # metrics about bookown distance reduced and rem distance reduced
        result['bookown_distance_not_covered_share'] = result['custom_metrics'][
            'bookown_distance_not_covered_share_mean']
        result['bookown_distance_not_covered'] = result['custom_metrics']['bookown_distance_not_covered_mean']
        result['distance_reduced_with_ownrides'] = result['custom_metrics']['distance_reduced_with_ownrides_mean']
        result['distance_reduced_with_shared'] = result['custom_metrics']['distance_reduced_with_shared_mean']
        result['distance_reduced_with_ownrides_share'] = result['custom_metrics'][
            'distance_reduced_with_ownrides_share_mean']
        result['distance_reduced_with_shared_share'] = result['custom_metrics'][
            'distance_reduced_with_shared_share_mean']
