# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)
        estat_actual = successor.get_agent_state(self.index)
        pos_actual = estat_actual.get_position()
        layout = game_state.data.layout
        width, height = layout.width, layout.height
        walls = layout.walls
        mid = width // 2
        if self.red:  
            enemy_positions = [(x, y) for x in range(mid, width) for y in range(height) if not walls[x][y]]
        else:  
            enemy_positions = [(x, y) for x in range(0, mid) for y in range(height) if not walls[x][y]]
        if len(food_list) > 0:
            min_distance = min([self.get_maze_distance(pos_actual, food) for food in food_list])
            features['distance_to_food'] = min_distance
        carrying = estat_actual.num_carrying
        distances = [self.get_maze_distance(pos_actual, pos) for pos in enemy_positions]
        dist_to_enemy_base = min(distances)
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = []
        scared_ghosts = []
        for e in enemies:
            pos_e = e.get_position()
            if pos_e is None:
                continue
            if not e.is_pacman:
                if getattr(e, 'scaredTimer', 0) > 0:
                    scared_ghosts.append(e)
                else:
                    ghosts.append(e)
        if ghosts and pos_actual is not None:
            manh_dists = [abs(pos_actual[0]-g.get_position()[0]) + abs(pos_actual[1]-g.get_position()[1]) for g in ghosts]
            min_ghost_dist = min(manh_dists)
            features['closest_ghost_distance'] = min_ghost_dist
            features['ghost_near'] = 1 if min_ghost_dist <= 5 else 0
        else:
            features['closest_ghost_distance'] = float('inf')
            features['ghost_near'] = 0
        if scared_ghosts and pos_actual is not None:
            manh_dists_s = [abs(pos_actual[0]-g.get_position()[0]) + abs(pos_actual[1]-g.get_position()[1]) for g in scared_ghosts]
            min_scared_dist = min(manh_dists_s)
            features['closest_scared_ghost_distance'] = min_scared_dist
            features['scared_ghost_near'] = 1 if min_scared_dist <= 5 else 0
            timers = [getattr(g, 'scaredTimer', 0) for g in scared_ghosts]
            near_timers = [t for t, d in zip(timers, manh_dists_s) if d <= 5]
            features['min_scared_timer'] = min(near_timers) if near_timers else 0
        else:
            features['closest_scared_ghost_distance'] = float('inf')
            features['scared_ghost_near'] = 0
            features['min_scared_timer'] = 0
        if carrying > 0:
            features['distance_to_start'] = dist_to_enemy_base
        else:
            features['distance_to_start'] = 0
        features['distance_to_enemy_base'] = dist_to_enemy_base
        return features

    def get_weights(self, game_state, action):
        features = self.get_features(game_state, action)
        dist_food = features.get('distance_to_food', float('inf'))
        dist_enemy = features.get('distance_to_enemy_base', float('inf'))
        ghost_near = features.get('ghost_near', 0)
        scared_near = features.get('scared_ghost_near', 0)
        estat = game_state.get_agent_state(self.index)
        carrying = estat.num_carrying
        if ghost_near:
            return {'successor_score': 40.0, 'distance_to_food': -0.5,
                    'distance_to_start': -5.0, 'distance_to_enemy_base': -2.0, 'ghost_near': -1000}
        min_scared_timer = features.get('min_scared_timer', 0)
        SCARED_LEAVE_THRESHOLD = 5
        MAX_CARRY = 5
        if scared_near:
            if carrying >= MAX_CARRY or min_scared_timer <= SCARED_LEAVE_THRESHOLD:
                return {'successor_score': 40.0, 'distance_to_food': -0.5,
                        'distance_to_start': -5.0, 'distance_to_enemy_base': -2.0}
            return {'successor_score': 90.0, 'distance_to_food': -4.0,
                    'distance_to_start': -0.5, 'distance_to_enemy_base': -0.5}
        if carrying == 0:
            base = {'successor_score': 100.0, 'distance_to_food': -1.5, 'distance_to_start': 0}
        elif carrying >= 3:
            base = {'successor_score': 40.0, 'distance_to_food': -0.5, 'distance_to_start': -5.0, 'distance_to_enemy_base': -2.0}
        elif dist_food < dist_enemy:
            base = {'successor_score': 60.0, 'distance_to_food': -3.0, 'distance_to_start': -1.0, 'distance_to_enemy_base': -0.5}
        else:
            base = {'successor_score': 40.0, 'distance_to_food': -0.5, 'distance_to_start': -5.0, 'distance_to_enemy_base': -2.0}
        if ghost_near:
            base['ghost_near'] = -1000
            base['closest_ghost_distance'] = 2.0
        return base

    def choose_action(self, game_state):
        state = game_state.get_agent_state(self.index)
        my_pos = state.get_position()
        if my_pos is None:
            return super().choose_action(game_state)
        if not state.is_pacman:
            enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
            pacmans = [e for e in enemies if e.is_pacman and e.get_position() is not None]
            if pacmans:
                pacmans.sort(key=lambda p: self.get_maze_distance(my_pos, p.get_position()))
                target = pacmans[0].get_position()
                return self._move_towards(game_state, target)
            return super().choose_action(game_state)
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [e for e in enemies if not e.is_pacman and e.get_position() is not None and getattr(e, 'scaredTimer', 0) == 0]
        scared_ghosts = [e for e in enemies if not e.is_pacman and getattr(e, 'scaredTimer', 0) > 0 and e.get_position() is not None]
        GHOST_RADIUS = 6
        close_ghosts = [g for g in ghosts if self.get_maze_distance(my_pos, g.get_position()) <= GHOST_RADIUS]
        if close_ghosts:
            capsules = self.get_capsules(game_state)
            nearest_capsule = min(capsules, key=lambda c: self.get_maze_distance(my_pos, c)) if capsules else None
            dist_home = self.get_maze_distance(my_pos, self.start)
            if nearest_capsule and self.get_maze_distance(my_pos, nearest_capsule) < dist_home:
                return self._move_towards(game_state, nearest_capsule)
            return self._move_towards(game_state, self.start)
        if scared_ghosts:
            carrying = state.num_carrying
            MAX_CARRY = 5
            if carrying < MAX_CARRY:
                food_list = self.get_food(game_state).as_list()
                if food_list:
                    food_list.sort(key=lambda f: self.get_maze_distance(my_pos, f))
                    return self._move_towards(game_state, food_list[0])
            return self._move_towards(game_state, self.start)
        food_list = self.get_food(game_state).as_list()
        if food_list:
            food_list.sort(key=lambda f: self.get_maze_distance(my_pos, f))
            return self._move_towards(game_state, food_list[0])
        return self._move_towards(game_state, self.start)

    def _move_towards(self, game_state, target):
        actions = game_state.get_legal_actions(self.index)
        best = None
        best_d = float('inf')
        for a in actions:
            if a == Directions.STOP:
                continue
            succ = self.get_successor(game_state, a)
            pos2 = succ.get_agent_state(self.index).get_position()
            if pos2 is None:
                continue
            d = self.get_maze_distance(pos2, target)
            if d < best_d:
                best_d = d
                best = a
        if best is None:
            return random.choice(actions)
        return best

class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    y_target = 4

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        if my_state.scared_timer > 0:
            features['on_defense'] = 0
        else:
            features['on_defense'] = 1 if not my_state.is_pacman else 0


        walls = game_state.get_walls()
        width = walls.width
        if self.red:
            border_x = width // 2 - 1
            target_x = border_x -3
        else:
            border_x = width // 2
            target_x = border_x +3
        target = (target_x, self.y_target)

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0 and my_state.scared_timer == 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        if len(invaders) == 0 and my_state.scared_timer == 0:
            features['target_patrol_dis'] = self.get_maze_distance(my_pos, target)
            if my_pos == target:
                self.y_target = 11 if self.y_target == 4 else 4
        else:
            features['target_patrol_dis'] = 0
        
        if my_state.scared_timer > 0:
            dangerous_enemies = [a for a in enemies if a.get_position() is not None]
            if len(dangerous_enemies) > 0:
                dists_de = [self.get_maze_distance(my_pos, a.get_position()) for a in dangerous_enemies]
                if min(dists_de)<5:
                    features['dangerous_enemy_dis'] = min(dists_de)
                else:
                    features['dangerous_enemy_dis'] = 0
            else:
                features['dangerous_enemy_dis'] = 0
        else:
            features['dangerous_enemy_dis'] = 0

        if my_state.scared_timer > 7:
            food_list = self.get_food(successor).as_list()
            features['successor_score'] = -len(food_list)
            if len(food_list) > 0:
                distances = [self.get_maze_distance(my_pos, f) for f in food_list]
                features['distance_to_food'] = min(distances)
            else:
                features['distance_to_food'] = 0
            # my_pos = successor.get_agent_state(self.index).get_position()
            # min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            # features['distance_to_food'] = min_distance
        else:
            if my_state.is_pacman:
                features['distance_to_food'] = abs(border_x - my_pos[0])
            else:
                features['distance_to_food'] = 0
        print(features)
        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -5, 'target_patrol_dis': -7, 'distance_to_food': -10, 'dangerous_enemy_dis': 1000, 'successor_score': 100}
