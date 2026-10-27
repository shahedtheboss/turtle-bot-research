import os
import time

import rclpy
from std_srvs.srv import Empty

from mini_dqn_nav.environment import RobotEnvironment
from mini_dqn_nav.agent import DQNAgent

import matplotlib.pyplot as plt
import numpy as np


def reset_simulation(node):
    '''reset gazebo between episodes'''

    for service_name in ("/reset_simulation", "/reset_world"):
        client = node.create_client(Empty, service_name)

        if client.wait_for_service(timeout_sec=1.0):
            
            future = client.call_async(Empty.Request())

            deadline = time.monotonic() + 5.0
            while rclpy.ok() and not future.done():
                if time.monotonic() >= deadline:
                    break
                rclpy.spin_once(node, timeout_sec=0.1)

            if future.done() and future.result() is not None:
                node.get_logger().info(
                    f"reset using {service_name}."
                    )
                return True

    node.get_logger().warning(
        "No usable Gazebo reset service found. Please make sure Gazebo is running."
    )
    return False

def main(args=None):
    rclpy.init(args=args)
    env = RobotEnvironment()
    agent = DQNAgent(state_size=7, action_size=5)

    episode_rewards = []
    episode_losses = []
    episode_steps = []
    episode_outcomes = []
    epsilons = []

    episodes = 120
    max_steps =100
    target_update_every = 2

    model_path = os.path.expanduser(
        "~/ros2_ws/src/mini_dqn_nav/mini_dqn_nav/models/dqn_model_5actions.pth"
    )
    try:
        #wait for initial lidar and odometry data
        deadline = time.monotonic() + 10.0

        while rclpy.ok() and env.get_state() is None:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "Timed out waiting for initial lidar and odometry data."
                )
            reason = "timeout"
            rclpy.spin_once(env, timeout_sec=0.1)

        for episode in range(1,episodes+1):
            state = env.get_state()
            episode_reward = 0.0
            losses = []
            ended = False
            mean_loss=0.0

            for step in range(max_steps):

                steps_taken = step + 1
                #select action using epsilon-greedy policy

                # action = agent.select_greedy_action(state)
                action = agent.select_action(state)
                #excecute action in the environment and observe the next state and reward
                next_state, reward, terminated, reason = env.step(action)

                #Store the experience in the replay memory and learn from a random sample of experiences
                done = terminated or (step == max_steps - 1)
                agent.remember(state, action, reward, next_state, done)
                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)
                state = next_state
                episode_reward += reward

                if terminated:
                    ended = True
                    env.get_logger().info(
                        f"Episode ended: {reason}"
                    )
                    break

            # These updates happen ONCE per episode.
            agent.decay_epsilon()

            # Calculate mean loss BEFORE recording it.
            mean_loss = (
                sum(losses) / len(losses)
                if losses
                else 0.0
            )

            episode_rewards.append(episode_reward)
            episode_losses.append(mean_loss)
            episode_steps.append(steps_taken)
            episode_outcomes.append(
                reason if ended else "timeout"
            )
            epsilons.append(agent.epsilon)

            if episode % target_update_every == 0:
                agent.update_target()

            # Print the episode summary ONCE, after its steps finish.
            env.get_logger().info(
                f"Episode {episode}/{episodes} | "
                f"reward={episode_reward:.2f} | "
                f"steps={steps_taken} | "
                f"epsilon={agent.epsilon:.3f} | "
                f"mean_loss={mean_loss:.4f} | "
                f"ended={ended}"
            )

            # Save progress after every episode.
            agent.save(model_path)

            # Reset the simulation before the next episode.
            if episode < episodes:
                if not reset_simulation(env):
                    env.get_logger().warning(
                        "Stopping training because the robot "
                        "cannot be reset safely."
                    )
                    break

                # Give the sensors time to publish after the reset.
                deadline = time.monotonic() + 2.0

                while rclpy.ok() and time.monotonic() < deadline:
                    rclpy.spin_once(env, timeout_sec=0.1)

                if env.get_state() is None:
                    raise RuntimeError(
                        "Sensor data unavailable after reset."
                    )

        env.get_logger().info(
            f"Training finished. Model saved to: {model_path}"
        )
        episodes_done = np.arange(1, len(episode_rewards) + 1)
        window = min(10, len(episode_rewards))

        moving_rewards = [
            np.mean(episode_rewards[max(0, i-window+1):i+1])
            for i in range(len(episode_rewards))
        ]

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle("TurtleBot3 DQN — Training Performance")

        axes[0, 0].plot(episodes_done, episode_rewards, label="Reward")
        axes[0, 0].plot(episodes_done, moving_rewards,
                        label=f"Moving average ({window})")
        axes[0, 0].set_title("Episode Reward")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        axes[0, 1].plot(episodes_done, episode_losses)
        axes[0, 1].set_title("Mean DQN Loss")
        axes[0, 1].grid(True)

        axes[1, 0].plot(episodes_done, epsilons)
        axes[1, 0].set_title("Epsilon Decay")
        axes[1, 0].grid(True)

        axes[1, 1].plot(episodes_done, episode_steps)
        axes[1, 1].set_title("Steps per Episode")
        axes[1, 1].grid(True)

        fig.tight_layout()
        fig.savefig("training_performance.png", dpi=150)
        plt.show()

    except KeyboardInterrupt:
        env.get_logger().info("Training interrupted.")

    except Exception as error:
        env.get_logger().error(
            f"Training failed: {error}"
        )
        raise

    finally:
        # Stop the robot before shutting down.
        if rclpy.ok():
            env.publish_stop()

        # Preserve the latest checkpoint.
        agent.save(model_path)

        env.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()