import argparse
import gymnasium as gym
import numpy as np
from crafter.wrappers import LinearReward
import imageio
from PIL import Image
from PIL import ImageFilter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--goal', type=str, default='explore', choices=[
       'mo', 'achievements', 'explore', 'kill_enemies', 'eat_cow', 'eat_plant'])

    args = parser.parse_args()
    
    env = gym.make('CrafterMOReward-v1', seed=args.seed, size=(84, 84))
    frames = []
    obs_frames = []

    # Define the linearization weights for multi-objective optimization
    if args.goal == 'mo':
        linearization_weights = None
    elif args.goal == 'achievements':
        linearization_weights = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    elif args.goal == 'explore':
        linearization_weights = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    elif args.goal == 'kill_enemies':
        linearization_weights = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    elif args.goal == 'eat_cow':
        linearization_weights = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    elif args.goal == 'eat_plant':
        linearization_weights = [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    else:
        raise ValueError(f'Unknown goal: {args.goal}')

    if linearization_weights is not None:
        env = LinearReward(env, weight=np.array(linearization_weights))

    # Reset the environment
    obs, info = env.reset()
    print("Initial Observation:", obs)
    done = False
    
    # Iterate through a few steps and record a video of the agent's actions
    for step in range(100):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)


        # Record the full observation frame (agent's input)
        obs_frame = obs  # shape (84,84,3)
        # ensure uint8 array
        if obs_frame.dtype != np.uint8:
            obs_frame = (obs_frame * 255).astype(np.uint8)
        obs_frames.append(np.array(obs_frame))

        # Capture raw frame and upscale to 256x256 with high-quality resampling
        raw_frame = env.render()
        # ensure uint8 array
        if raw_frame.dtype != np.uint8:
            raw_frame = (raw_frame * 255).astype(np.uint8)
        img = Image.fromarray(raw_frame)
        # upscale using Lanczos filter for smoothness
        img = img.resize((256, 256), resample=Image.LANCZOS)
        # apply slight sharpening to enhance edges
        img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=1))
        frames.append(np.array(img))

        # Print the step information
        done = terminated or truncated
        print(f"Step {step + 1}: Action: {action}, Reward: {reward}, Observation: {obs}")
        
        if done:
            print("Episode finished.")
            break

    # Save recorded frames as a GIF
    imageio.mimsave('crafter_render.gif', frames, fps=10)
    print("Saved GIF to crafter_render.gif")
    # Save observation frames as a separate GIF
    imageio.mimsave('crafter_observations.gif', obs_frames, fps=10)
    print("Saved GIF to crafter_observations.gif")
