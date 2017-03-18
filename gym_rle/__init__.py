import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)


for game in ['aladdin', 'bust_a_move', 'classic_kong', 'final_fight', 'f_zero', 'gradius_iii',
             'mortal_kombat', 'nba_give_n_go', 'super_mario_all_stars',
             'super_mario_world', 'street_fighter_ii', 'tetris_and_dr_mario', 'wolfenstein']:
        for obs_type in ['image', 'ram']:
            name = ''.join([g.capitalize() for g in game.split('_')])
            if obs_type == 'ram':
                name = '{}-ram'.format(name)
            nondeterministic = False

            register(
                id='{}-v0'.format(name),
                entry_point='gym_rle.envs:RleEnv',
                kwargs={'game': game, 'obs_type': obs_type, 'repeat_action_probability': 0.25},
                tags={'wrapper_config.TimeLimit.max_episode_steps': 10000},
                nondeterministic=nondeterministic,
            )
