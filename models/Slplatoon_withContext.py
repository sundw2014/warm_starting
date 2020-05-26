import numpy as np
import sys
sys.path.append('..')
from utils.general_utils import llist_generator
from llist import dllist, dllistnode
from NiMC import NiMC
from CMAB import ContextualMAB

__all__ = ['Slplatoon_withContext']

prob_close = np.array([0.7,0.15,0.15]) # prob for brake/cruise/speedup when close
prob_fine = np.array([0.15,0.7,0.15])
prob_far = np.array([0.15,0.15,0.7])
v_brake = 1
v_cruise = 4
v_speedup = 7
v_error = 0.1
velocities = [v_brake, v_cruise, v_speedup]

threshold_close = 3
threshold_far = 5

unsafe_rule_close = 1

num_cars = 4
initial_separation_between_cars = 10
range_of_initial_set_per_car = 5
Theta = [range(n*initial_separation_between_cars, n*initial_separation_between_cars+range_of_initial_set_per_car+1) for n in range(num_cars)]
Theta = np.meshgrid(*Theta)

class Slplatoon_nimc(NiMC):
    def __init__(self, k=11):
        super(Slplatoon_nimc, self).__init__()
        self.set_Theta(Theta)
        self.set_k(k)

        self.actions = [[] for _ in range(num_cars)]

    def set_conext(self, context):
        velocities[0] = context[0]
        self.reaction_time = context[1]

    def is_unsafe(self, state):
        for i in range(1, len(state)):
            if state[i-1] - state[i] < unsafe_rule_close:
                return True
        return False

    def transition(self, state):
        assert len(state) == num_cars
        state = np.array(state)

        state_current = dllist(state)
        state_old = state_current
        state_current = dllist()

        # update the state
        for car in llist_generator(state_old):
            ita = np.random.randn() * v_error
            #print(ita)
            if car is state_old.first:
                # first car, always cruise
                position = car.value + v_cruise + ita
                state_current.append(position)
                # step_desc += 'first'
            else:
                # not the first car
                distance = car.prev.value - car.value
                p = prob_close if distance < threshold_close else prob_fine if distance < threshold_far else prob_far
                self.actions[i].append(p)

                if len(self.actions[i]) > self.reaction_time+1:
                    p = self.actions[i][-int(self.reaction_time)-1]

                v = np.random.choice(velocities, p = p)
                position = car.value + v# + ita
                state_current.append(position)
        state = list(state_current)
        return state

class Slplatoon_withContext(ContextualMAB):
    """docstring for Slplatoon_withContext."""
    def __init__(self):
        super(Slplatoon_withContext, self).__init__()
        self.nimc = Slplatoon_nimc()
        self.nU = 4
        self.context_map = [[1,], [0, 1, 2, 3]]
        self.set_constants(nArms = len(self.nimc.Theta), nZ = 1, nU = self.nU)

    def sample_context(self):
        context = [0, np.random.randint(self.nU)]
        return context

    def play_context(self, arm, context):
        context = [self.context_map[i][context[i]] for i in len(context)]
        self.nimc.set_conext(context)
        return self.nimc(nimc.Theta[arm,:])
