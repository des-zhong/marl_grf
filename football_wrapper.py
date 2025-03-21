import gfootball.env as football_env
from baselines import logger
import os

level = 0.05
from baselines.bench import monitor
def create_single_football_env(iprocess, FLAGS):
  global levels
  """Creates gfootball environment."""
  env = football_env.create_environment(
      env_name='11_vs_11_'+level, stacked=('stacked' in FLAGS.state),
      # number_of_left_players_agent_controls = 5,
      rewards=FLAGS.reward_experiment,
      logdir=logger.get_dir(),
      write_goal_dumps=FLAGS.dump_scores and (iprocess == 0),
      write_full_episode_dumps=FLAGS.dump_full_episodes and (iprocess == 0),
      render=FLAGS.render and (iprocess == 0),
      dump_frequency=50 if FLAGS.render and iprocess == 0 else 0)
  env = monitor.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(),
                                                               str(iprocess)))
  return env

class 