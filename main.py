from Game.game_window import GameWindow
from Brain.brain import Brain
import random
import time
import pyglet
import numpy as np


'''Params begin'''
# To learn or just play
learn = True

# Filename to save the model during learning
filename = "Models/test"

# After how many generation should be the model saved.
save_per_generation = 2

# To show what does the species do in a population
# (Slows down the learning, but not that much)
show_population = True

# Number of species per population
species_per_population = 1000

# How many fittest species could survive
# without reproduction
number_of_fit_to_survive = 5

# After how many does should the running stop
number_of_generations = 10000

# What the AI can see
# Light rays from the center of the car are projected in
# different directions. This array hold the directions.
# 0 points forward -180 or 180 backwards
# You can play with it :)
ray_angles = [-180, -135, -90, -60, -30, 0, 30, 60, 90, 135]


# What moves are allowed for the AI to take.
# Keep in mind, if you don't add a forward motion, it can spin around...
# A motion consists of the following vectors: [up, left, right]
# E.g.: [1,0,0] means straight forward motion
action2movetable = [[1, 1, 0], [1, 0, 0], [1, 0, 1]]

# After how many episode, should be the track regenerated
# -1 means never
track_regeneration_per_episode = -1

'''Params end'''

# If we don't learn, population should be shown anyway
if not learn:
    show_population = True

# If we don't learn, we only want to see one car (the fittest species
# from learning)
species_per_population = species_per_population if learn else 1


class MyGame(GameWindow):
    """Overrided GameWindow instance
    You need to do this, to specify what happens during update
    """
    def __init__(self, update_screen, width, height):
        super().__init__(update_screen, width, height)

        # Save input variables
        self.brain = None
        self.learn = False
        self.update_screen = update_screen

        # Additional required variables
        self.fitness_sum = 0
        self.best_reward = 0
        self.states = []

        # Labels
        self.epi_label = pyglet.text.Label(text="Generation: 1", x=10, y=35, batch=self.batches['main'],
                                           group=self.subgroups['base'])

        self.reward_label = pyglet.text.Label(text="Best reward: 0", x=10, y=20, batch=self.batches['main'],
                                              group=self.subgroups['base'])

    # Main function
    def update(self):
        """Engines of the game
        It updates and moves everything
        You specify here, what you want to do, in each timestep
        """
        # if self.time is 0:
        self.states = self.get_states()
        actions = self.brain.act(self.states)
        # self.states = []
        for idx, car in enumerate(self.get_cars()):
            if not car.died and not car.won:
                reward, new_state, done = car.step(actions[idx])
                # self.states.append(new_state)

        # self.states = np.reshape(self.states, [1, len(self.states)])

        if self.is_game_ended():
            if self.episode % save_per_generation == 0 and self.learn:
                self.save_brain_of_fittest()
            if self.learn:
                self.natural_selection()
                self.mutate_babies()
            self.episode += 1
            self.epi_label.text = "Generation: {0}".format(self.episode)
            self.reward_label.text = "Best reward: {0}".format(self.best_reward)
            if track_regeneration_per_episode is not -1 and self.episode % track_regeneration_per_episode == 0:
                self.track.GenTrack()
            self.reset()

    # Helper functions
    def get_states(self):
        """Returns with all the states in an array
        :return states array
        """
        states = np.array([car.get_state() for car in self.get_cars()]).flatten()
        return np.reshape(states, [1, len(states)])

    def set_brain(self, brain):
        """Sets the brain for the cars
        :argument brain: the brain to set"""
        self.brain = brain

    def set_learning(self):
        """Set game to learning
        Note: by default, the game is not learning"""
        self.learn = True

    # Genetic Algorithms functions
    def natural_selection(self):
        """Selects the fittest species from the population
        """
        self._calculate_fitness_sum(self.get_cars())
        fathers = []
        mothers = []

        # Select best species
        best_species = self.select_best_species(self.get_cars(), number_of_fit_to_survive)

        best_car = self.get_cars([best_species[0],best_species[0]+1])[0]

        # Set best reward
        self.best_reward = best_car.last_reward

        print("Generation {0} best fitness {1:0F}".format(self.episode,best_car.last_reward))

        fathers += best_species
        mothers += best_species

        # Reproduction
        number_to_generate = self.species_per_population - number_of_fit_to_survive
        parents = self._select_parents(self.get_cars(), number_to_generate)
        keys = list(parents.keys())
        fathers += list({k:parents[k] for k in keys[:number_to_generate]}.values())
        mothers += list({k:parents[k] for k in keys[number_to_generate:2*number_to_generate]}.values())
        self.brain.generate_new_weights(fathers,mothers)

    def select_best_species(self,cars,number_of_best_species_to_select=1):
        """Selects best species from the population
        :argument cars: to select from
        :argument number_of_best_species_to_select: how many best species to select
        :return best species array"""
        self._sort_population_by_fitness(cars)
        best_car_ids =[car.id for car in cars[:number_of_best_species_to_select]]
        return best_car_ids

    def mutate_babies(self):
        """Mutate the babies of the parents
        """
        wons = [car.won for car in self.get_cars()]
        self.brain.mutate(wons)

    def save_brain_of_fittest(self):
        """Saves the brain of the fittest
        """
        cars = self.get_cars([1, species_per_population + 1])
        fittest = self.select_best_species(cars)
        brain.save_model_by_id(fittest)

    # Helper functions
    def _sort_population_by_fitness(self, cars):
        """Sort the population by reward
        """
        cars.sort(key=lambda car: car.last_reward,reverse=True)

    def _calculate_fitness_sum(self, cars):
        """Calculate the sum of the rewards
        """
        self.fitness_sum = 0
        for car in cars:
            self.fitness_sum += max(0,car.last_reward)

    def _select_parents(self, cars, number_of_parents):
        """Select parents for breeding
        :argument cars: to select from
        :argument number_of_parents: how many parents to select
        Note: it will select this many mother and father
        :return parents array"""
        rands = [random.uniform(0, self.fitness_sum) for _ in range(number_of_parents * 2)]
        parents = {}
        dists = np.ones([number_of_parents * 2]) * 10000

        running_sum = 0
        for idx, car in enumerate(cars):
            running_sum += max(0, car.last_reward)
            for idy, rand in enumerate(rands):
                dist = running_sum - rand
                if 0 < dist < dists[idy]:
                    dists[idy] = dist
                    parents[idy] = idx
        return parents


# The brain instance, which will compute what to do for a state
# In this case, every cars "brain" is added to a big module, for sake of speed,
# this way, the model has to be loaded once per step per frame
# Disadvantage: if there are less cars to step with, it will use the big model either
# so it does not matter, if only 10 cars alive from 500 or 500, the action calculation
# step will take the same time...
brain = Brain(len(ray_angles), action2movetable, species_per_population, learn, filename)

# Creating the Game instance
game = MyGame(show_population, 800, 600)
game.set_brain(brain)
if learn:
    game.set_learning()

# Generate track
game.generate_track(draw_gates=False)

# Set player number
game.set_car_number(species_per_population,ray_angles, lapse_until_won=5)

# Set length of the game
game.set_game_length(number_of_generations)

# Resetting the environment
game.reset()

# Run the game
if __name__ == "__main__":
    game.run()


