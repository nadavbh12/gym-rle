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

            register(
                id='{}-v4'.format(name),
                entry_point='gym.envs.atari:AtariEnv',
                kwargs={'game': game, 'obs_type': obs_type},
                max_episode_steps=100000,
                nondeterministic=nondeterministic,
            )

            frameskip = 4
            # Use a deterministic frame skip.
            register(
                id='{}Deterministic-v0'.format(name),
                entry_point='gym_rle.envs:RleEnv',
                kwargs={'game': game, 'obs_type': obs_type, 'frameskip': frameskip, 'repeat_action_probability': 0.25},
                max_episode_steps=100000,
                nondeterministic=nondeterministic,
            )

            register(
                id='{}Deterministic-v4'.format(name),
                entry_point='gym_rle.envs:RleEnv',
                kwargs={'game': game, 'obs_type': obs_type, 'frameskip': frameskip},
                max_episode_steps=100000,
                nondeterministic=nondeterministic,
            )

            register(
                id='{}NoFrameskip-v0'.format(name),
                entry_point='gym_rle.envs:RleEnv',
                kwargs={'game': game, 'obs_type': obs_type, 'frameskip': 1, 'repeat_action_probability': 0.25}, # A frameskip of 1 means we get every frame
                max_episode_steps=frameskip * 100000,
                nondeterministic=nondeterministic,
            )

            # No frameskip. (Atari has no entropy source, so these are
            # deterministic environments.)
            register(
                id='{}NoFrameskip-v4'.format(name),
                entry_point='gym_rle.envs:RleEnv',
                kwargs={'game': game, 'obs_type': obs_type, 'frameskip': 1},
                # A frameskip of 1 means we get every frame
                max_episode_steps=frameskip * 100000,
                nondeterministic=nondeterministic,
            )
