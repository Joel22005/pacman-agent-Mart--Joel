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
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        # ADAPTAR AL CAS QUE SIGUIS L'EQUIP BLAU
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.get_score(successor)
        estat_actual = successor.get_agent_state(self.index)
        pos_actual = estat_actual.get_position()
        layout = game_state.data.layout
        width, height = layout.width, layout.height
        walls = layout.walls
        mid = width // 2
        red_positions = [(x, y) for x in range(0, mid) for y in range(height) if not walls[x][y]]
        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            try:
                min_distance = min([self.get_maze_distance(pos_actual, food) for food in food_list])
            except Exception:
                min_distance = float('inf')
            features['distance_to_food'] = min_distance
        carrying = estat_actual.num_carrying
        distances = []
        try:
            distances = [self.get_maze_distance(pos_actual, pos) for pos in red_positions]
            dist_to_red = min(distances)
        except Exception:
            dist_to_red = float('inf')
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
            manh_dists = [abs(pos_actual[0] - g.get_position()[0]) + abs(pos_actual[1] - g.get_position()[1]) for g in ghosts]
            min_ghost_dist = min(manh_dists)
            features['closest_ghost_distance'] = min_ghost_dist
            features['ghost_near'] = 1 if min_ghost_dist <= 5 else 0
        else:
            features['closest_ghost_distance'] = float('inf')
            features['ghost_near'] = 0
        if scared_ghosts and pos_actual is not None:
            manh_dists_s = [abs(pos_actual[0] - g.get_position()[0]) + abs(pos_actual[1] - g.get_position()[1]) for g in scared_ghosts]
            min_scared_dist = min(manh_dists_s)
            features['closest_scared_ghost_distance'] = min_scared_dist
            features['scared_ghost_near'] = 1 if min_scared_dist <= 5 else 0
            # compute min scared timer among scared ghosts that are within the near radius
            timers = [getattr(g, 'scaredTimer', 0) for g in scared_ghosts]
            near_timers = [t for t, d in zip(timers, manh_dists_s) if d <= 5]
            features['min_scared_timer'] = min(near_timers) if near_timers else 0
        else:
            features['closest_scared_ghost_distance'] = float('inf')
            features['scared_ghost_near'] = 0
            features['min_scared_timer'] = 0
        if carrying > 0 and self.start is not None: #Si hem agafat algún menjar calcula la distància a la base
            features['distance_to_start'] = dist_to_red
        else:
            features['distance_to_start'] = 0
        features['distance_to_red'] = dist_to_red
        return features

    def get_weights(self, game_state, action):
        features = self.get_features(game_state, action)
        dist_food = features.get('distance_to_food', float('inf'))
        dist_red = features.get('distance_to_red', float('inf'))
        ghost_near = features.get('ghost_near', 0)
        closest_ghost = features.get('closest_ghost_distance', float('inf'))
        scared_near = features.get('scared_ghost_near', 0)
        closest_scared = features.get('closest_scared_ghost_distance', float('inf'))
        estat = game_state.get_agent_state(self.index)
        carrying = estat.num_carrying
        # If there is a nearby non-scared ghost, avoid immediately (safety first)
        if ghost_near:
            return {'successor_score': 40.0, 'distance_to_food': -0.5, 'distance_to_start': -5.0, 'distance_to_red': -2.0, 'ghost_near': -1000}

        # Exploit scared ghosts: collect as much as possible while scared, but
        # stop collecting and return when the scared timer is about to expire
        min_scared_timer = features.get('min_scared_timer', 0)
        SCARED_LEAVE_THRESHOLD = 5  # when timer <= this, stop collecting and return
        MAX_CARRY = 3
        if scared_near:
            # If carrying too much or timer low, return to base
            if carrying >= MAX_CARRY or min_scared_timer <= SCARED_LEAVE_THRESHOLD:
                return {'successor_score': 40.0, 'distance_to_food': -0.5, 'distance_to_start': -5.0, 'distance_to_red': -2.0}
            # Otherwise, exploit: strongly prefer getting more food while scared
            return {'successor_score': 90.0, 'distance_to_food': -4.0, 'distance_to_start': -0.5, 'distance_to_red': -0.5}
        if carrying == 0:
            base = {'successor_score': 100.0, 'distance_to_food': -1.5, 'distance_to_start': 0}
        elif carrying >= 3:
            base = {'successor_score': 40.0, 'distance_to_food': -0.5, 'distance_to_start': -5.0, 'distance_to_red': -2.0}
        elif dist_food < dist_red:
            base = {'successor_score': 60.0, 'distance_to_food': -3.0, 'distance_to_start': -1.0, 'distance_to_red': -0.5}
        else:
            base = {'successor_score': 40.0, 'distance_to_food': -0.5, 'distance_to_start': -5.0, 'distance_to_red': -2.0}
        if ghost_near:
            base['ghost_near'] = -1000
            base['closest_ghost_distance'] = 2.0
        return base

    def choose_action(self, game_state):
        """Improved offensive action selection.

        - If not Pacman: fallback to default.
        - If carrying and base is closer than nearest food: return to base.
        - If nearby non-scared ghosts detected: pick the safest food (safety margin)
          and choose the action that moves toward it while maximizing distance to ghosts.
        - Otherwise fall back to default evaluation-based action.
        """
        state = game_state.get_agent_state(self.index)
        my_pos = state.get_position()

        if my_pos is None:
            return super().choose_action(game_state)

        # Only apply aggressive safety routing when we're Pacman
        if not state.is_pacman:
            return super().choose_action(game_state)

        # If carrying food, compare distance to base vs nearest food
        carrying = state.num_carrying
        if carrying > 0:
            foods_check = self.get_food(game_state).as_list()
            if foods_check:
                try:
                    foods_check.sort(key=lambda f: self.get_maze_distance(my_pos, f))
                except Exception:
                    pass
                nearest_food = foods_check[0]
                try:
                    dist_food_nearest = self.get_maze_distance(my_pos, nearest_food)
                except Exception:
                    dist_food_nearest = float('inf')
            else:
                dist_food_nearest = float('inf')
            try:
                dist_to_base = self.get_maze_distance(my_pos, self.start) if self.start is not None else float('inf')
            except Exception:
                dist_to_base = float('inf')
            if dist_to_base <= dist_food_nearest:
                # move toward base
                actions = game_state.get_legal_actions(self.index)
                rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
                best_act = None
                best_d = float('inf')
                for a in actions:
                    if a == Directions.STOP:
                        continue
                    succ = self.get_successor(game_state, a)
                    pos2 = succ.get_agent_state(self.index).get_position()
                    if pos2 is None:
                        continue
                    try:
                        d = self.get_maze_distance(pos2, self.start)
                    except Exception:
                        d = float('inf')
                    if d < best_d or (d == best_d and a != rev and best_act == rev):
                        best_d = d
                        best_act = a
                if best_act is not None:
                    return best_act

        # Detect nearby non-scared ghosts
        ghosts = []
        for opp in self.get_opponents(game_state):
            e = game_state.get_agent_state(opp)
            pos_e = e.get_position()
            if pos_e is None:
                continue
            if not e.is_pacman and getattr(e, 'scaredTimer', 0) <= 0:
                ghosts.append(pos_e)

        DANGER_RADIUS = 7
        nearby = [g for g in ghosts if self.get_maze_distance(my_pos, g) <= DANGER_RADIUS]
        if not nearby:
            return super().choose_action(game_state)

        # select candidate foods and evaluate safety
        foods = self.get_food(game_state).as_list()
        if not foods:
            return super().choose_action(game_state)
        try:
            foods.sort(key=lambda f: self.get_maze_distance(my_pos, f))
        except Exception:
            pass
        FOOD_CANDIDATES = 6
        candidates = foods[:FOOD_CANDIDATES]

        best_food = None
        best_safety = -1e9
        for f in candidates:
            try:
                agent_d = self.get_maze_distance(my_pos, f)
            except Exception:
                agent_d = float('inf')
            ghost_ds = []
            for g in nearby:
                try:
                    gd = self.get_maze_distance(g, f)
                except Exception:
                    gd = float('inf')
                ghost_ds.append(gd)
            if not ghost_ds:
                continue
            safety = min(ghost_ds) - agent_d
            if safety > best_safety:
                best_safety = safety
                best_food = f

        if best_food is None:
            return super().choose_action(game_state)

        # score actions by moving toward best_food and away from ghosts
        actions = game_state.get_legal_actions(self.index)
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        best_actions = []
        best_score = None
        for a in actions:
            if a == Directions.STOP:
                continue
            succ = self.get_successor(game_state, a)
            pos2 = succ.get_agent_state(self.index).get_position()
            if pos2 is None:
                continue
            try:
                d_food = self.get_maze_distance(pos2, best_food)
            except Exception:
                d_food = float('inf')
            try:
                min_ghost = min([self.get_maze_distance(pos2, g) for g in nearby])
            except Exception:
                min_ghost = 0

            score = -d_food + 0.8 * min_ghost
            if min_ghost <= 2:
                score -= 50
            try:
                agent_d_food = self.get_maze_distance(my_pos, best_food)
                min_gd_food = min([self.get_maze_distance(g, best_food) for g in nearby])
            except Exception:
                agent_d_food = float('inf')
                min_gd_food = float('inf')
            if min_gd_food <= agent_d_food + 1:
                score -= 30

            if best_score is None or score > best_score[0] or (score == best_score[0] and min_ghost > best_score[1]):
                best_score = (score, min_ghost)
                best_actions = [a]
            elif score == best_score[0] and min_ghost == best_score[1]:
                best_actions.append(a)

        if not best_actions:
            return super().choose_action(game_state)
        non_rev = [a for a in best_actions if a != rev]
        if non_rev:
            return random.choice(non_rev)
        return random.choice(best_actions)


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
