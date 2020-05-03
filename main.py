# Train
import gym
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from policy_net import Policy_net
from ppo import PPOTrain

EPISODES = int(1e5)
GAMMA = 0.95


def main():
    env = gym.make('CartPole-v0')
    env.seed(0)
    ob_space = env.observation_space
    Policy = Policy_net('policy', env)
    Old_Policy = Policy_net('old_policy', env)
    PPO = PPOTrain(Policy, Old_Policy, gamma=GAMMA)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./log/train', sess.graph)
        sess.run(tf.global_variables_initializer()) # Inicializa as redes

        obs = env.reset()   # Reseta o ambiente e obtem a primeira observaçao
        reward = 0          # Armazena as recompensas
        success_num = 0

        for iteration in range(EPISODES):  # Loop do episodio
            observations = []   # Array pra armazenar as observaçoes
            actions = []        # Array pra armazenar as açoes
            v_preds = []        # Array pra armazenar as previsoes
            rewards = []        # Array pra armazenar as recompensas
            run_policy_steps = 0# Contador de passos em cada epsodio
            env.render()        # Renderiza o ambiente
            while True:         # run policy RUN_POLICY_STEPS which is much less than episode length
                run_policy_steps += 1   # Incrementa contador de passos de cada ep
                obs = np.stack([obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs
                act, v_pred = Policy.act(obs=obs, stochastic=True)

                act     = act.item()
                v_pred  = v_pred.item()

                observations.append(obs)
                actions.append(act)
                v_preds.append(v_pred)
                rewards.append(reward)

                next_obs, reward, done, info = env.step(act)

                if done:
                    v_preds_next = v_preds[1:] + [0]  # next state of terminate state has 0 state value
                    obs = env.reset()
                    reward = -1
                    break
                else:
                    obs = next_obs

            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_length', simple_value=run_policy_steps)])
                               , iteration)
            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_reward', simple_value=sum(rewards))])
                               , iteration)

            # end condition of test
            if sum(rewards) >= 195:
                success_num += 1
                if success_num >= 100:
                    saver.save(sess, './model/model.ckpt')
                    print('Clear!! Model saved.')
                    break
            else:
                success_num = 0

            gaes = PPO.get_gaes(rewards=rewards, v_preds=v_preds, v_preds_next=v_preds_next)

            # convert list to numpy array for feeding tf.placeholder
            observations = np.reshape(observations, newshape=[-1] + list(ob_space.shape))
            actions = np.array(actions).astype(dtype=np.int32)
            rewards = np.array(rewards).astype(dtype=np.float32)
            v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)
            gaes = np.array(gaes).astype(dtype=np.float32)
            gaes = (gaes - gaes.mean()) / gaes.std()

            PPO.assign_policy_parameters()

            inp = [observations, actions, rewards, v_preds_next, gaes]

            # train
            for epoch in range(4):
                sample_indices = np.random.randint(low=0, high=observations.shape[0], size=64)  # indices's are in [low, high)
                sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                PPO.train(obs=sampled_inp[0],
                    actions=sampled_inp[1],
                    rewards=sampled_inp[2],
                    v_preds_next=sampled_inp[3],
                    gaes=sampled_inp[4])

            summary = PPO.get_summary(obs=inp[0],
                actions=inp[1],
                rewards=inp[2],
                v_preds_next=inp[3],
                gaes=inp[4])[0]

            writer.add_summary(summary, iteration)
        writer.close()


if __name__ == '__main__':
    main()
