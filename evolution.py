from player import Player
import numpy as np
from config import CONFIG
import copy, random
import pickle
class Evolution():
    num_genes = 1

    def __init__(self, mode):
        self.mode = mode

    # calculate fitness of players
    def calculate_fitness(self, players, delta_xs):
        for i, p in enumerate(players):
            p.fitness = delta_xs[i]

    def add_gaussian_noise(self, player:Player, mean:float, standard_deviation:float):
        player.nn.first_layer_weights_matrix += np.random.normal(mean,standard_deviation,player.nn.first_layer_weights_matrix.shape) 
        player.nn.second_layer_weights_matrix += np.random.normal(mean,standard_deviation,player.nn.second_layer_weights_matrix.shape)
        player.nn.b1 += np.random.normal(mean,standard_deviation,player.nn.b1.shape)
        player.nn.b2 += np.random.normal(mean,standard_deviation,player.nn.b2.shape)
        return player

    def mutate(self, child):
        if 0.8 >= np.random.uniform(0,1):
            return self.add_gaussian_noise(child, 0.0,0.3)
        else:
            return child


    def q_tournement(self,players,  num_players, q):
        chosen_list = []
        for x in range(num_players):
            candidates = [players[x] for x in random.sample(range(num_players), q)]
            chosen_list.append(copy.deepcopy(sorted(candidates, key=lambda x: x.fitness, reverse=True)[0]))

        return chosen_list

      
    def crossover(self, parent_players, num_players):
        children = []
        w1_row_num = parent_players[0].nn.first_layer_weights_matrix.shape[0]
        w2_row_num = parent_players[0].nn.second_layer_weights_matrix.shape[0]
        b1_shape = parent_players[0].nn.b1.shape[0]
        b2_shape = parent_players[0].nn.b2.shape[0]
        index = 0
        for i in range(num_players//2):
            parent1 = parent_players[index]
            parent2 = parent_players[index+1]
            child1 = Player('helicopter')
            child2 = Player('helicopter')
            child1.nn.first_layer_weights_matrix = np.concatenate((parent1.nn.first_layer_weights_matrix[:w1_row_num//2], parent2.nn.first_layer_weights_matrix[w1_row_num//2:]), axis = 0)
            child2.nn.first_layer_weights_matrix = np.concatenate((parent2.nn.first_layer_weights_matrix[:w1_row_num//2], parent1.nn.first_layer_weights_matrix[w1_row_num//2:]), axis = 0)
            child2.nn.second_layer_weights_matrix = np.concatenate((parent1.nn.second_layer_weights_matrix[:w2_row_num//2], parent2.nn.second_layer_weights_matrix[w2_row_num//2:]), axis=0)
            child2.nn.second_layer_weights_matrix = np.concatenate((parent2.nn.second_layer_weights_matrix[:w2_row_num//2], parent1.nn.second_layer_weights_matrix[w2_row_num//2:]), axis = 0)
            child1.nn.b1 = np.concatenate((parent1.nn.b1[:b1_shape//2],parent2.nn.b1[b1_shape//2:]), axis = 0)
            child2.nn.b1 = np.concatenate((parent2.nn.b1[:b1_shape//2],parent1.nn.b1[b1_shape//2:]), axis = 0)
            child1.nn.b2 = np.concatenate((parent1.nn.b2[:b2_shape//2],parent2.nn.b2[b2_shape//2:]), axis = 0)
            child2.nn.b2 = np.concatenate((parent2.nn.b2[:b2_shape//2],parent1.nn.b2[b2_shape//2:]), axis = 0)
            children.extend([child1, child2])
            index += 2
        return children

    # box = 2standar_deviation = 0.3 , q= 5 is best for normal situation in helicopter, gravity
    # box =2 standar_deviation = 0.3 , q= 3 is best for crosover in helicopter
    # box =1, standar_deviation = 0.3 , q= 10 is best for crossover in gravity
    # box =1, standar_deviation = 0.3 , q= 10 is best for crossover in airplane
    # box =1, standar_deviation = 0.3 , q= 10 is best for crossover in helicopter
    def generate_new_population(self, num_players, prev_players=None):

        # in first generation, we create random players
        if prev_players is None:
            return [Player(self.mode) for _ in range(num_players)]

        else:

            # TODO
            # num_players example: 150
            # prev_players: an array of `Player` objects
            # parents = self.roulette_wheel_selection(prev_players, num_players)

            # TODO (additional): a selection method other than `fitness proportionate`
            # parents = self.q_tournement(prev_players,num_players,5)

            # TODO (additional): implementing crossover
            parents = self.q_tournement(prev_players,num_players,10)
            parents_copy = copy.deepcopy(parents)
            parents_copy = self.crossover(parents_copy, num_players)
            
            # parents_copy = copy.deepcopy(parents)
            new_players = [self.mutate(x) for x in parents_copy]
            return new_players

    def roulette_wheel_selection(self,players,  num_players):
        population_fitness = sum([p.fitness for p in players])
        player_probabilities = [p.fitness/population_fitness for p in players]
        return list(np.random.choice(players,num_players, p=player_probabilities))

    def next_population_selection(self, players, num_players):
        #plotting
        if self.num_genes == 1:
            generations_data = { "min":[], "max":[], "avg":[]}
        else :
            readed = open('generations_data.obj', 'rb') 
            generations_data = pickle.load(readed)
            
        players_fitness = [p.fitness for p in players]
        generations_data['min'].append(min(players_fitness))
        generations_data['max'].append(max(players_fitness))
        generations_data['avg'].append(sum(players_fitness)/len(players_fitness))
        self.num_genes += 1
        
        generations_data_file = open('generations_data.obj', 'wb') 
        pickle.dump(generations_data, generations_data_file)
        generations_data_file.close()

        # num_players example: 100
        # players: an array of `Player` objects
        # return players[: num_players]
        # a selection method other than `top-k`
        return self.roulette_wheel_selection(players, num_players)
        
        

    