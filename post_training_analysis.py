import matplotlib.pyplot as plt
import numpy as np
import torch

def analyze_training_results(agent):
    print("=== TRAINING ANALYSIS AFTER 1000 EPISODES ===")
    
    stats = agent.get_stats()
    
    print(f"Current Statistics:")
    print(f"  Average Reward (last 100 episodes): {stats['avg_reward']:.2f}")
    print(f"  Max Reward (last 100 episodes): {stats['max_reward']:.2f}")
    print(f"  Average Episode Length: {stats['avg_length']:.1f}")
    print(f"  Current Epsilon: {stats['epsilon']:.4f}")
    print(f"  Average Loss: {stats['avg_loss']:.4f}")
    print(f"  Current Learning Rate: {stats['lr']:.6f}")
    
    if len(agent.rewards_history) >= 100:
        recent_rewards = agent.rewards_history[-100:]
        early_rewards = agent.rewards_history[:100]
        
        improvement = np.mean(recent_rewards) - np.mean(early_rewards)
        print(f"\nReward Improvement: {improvement:.2f}")
        
        if improvement > 0:
            print("✅ Model is improving!")
        else:
            print("❌ Model might need adjustments")
    
    if len(agent.rewards_history) >= 50:
        recent_50 = agent.rewards_history[-50:]
        std_recent = np.std(recent_50)
        print(f"Recent Reward Stability (std): {std_recent:.2f}")
        
        if std_recent < 5.0:
            print("✅ Training is stable")
        else:
            print("❌ Training is unstable")
    
    return stats

def plot_training_progress(agent, save_path='training_progress.png'):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    if agent.rewards_history:
        axes[0,0].plot(agent.rewards_history)
        axes[0,0].set_title('Episode Rewards')
        axes[0,0].set_xlabel('Episode')
        axes[0,0].set_ylabel('Reward')
        axes[0,0].grid(True)
        
        if len(agent.rewards_history) >= 100:
            moving_avg = np.convolve(agent.rewards_history, np.ones(100)/100, mode='valid')
            axes[0,0].plot(range(99, len(agent.rewards_history)), moving_avg, 'r-', label='Moving Average (100)')
            axes[0,0].legend()
    
    if agent.episode_lengths:
        axes[0,1].plot(agent.episode_lengths)
        axes[0,1].set_title('Episode Lengths')
        axes[0,1].set_xlabel('Episode')
        axes[0,1].set_ylabel('Steps')
        axes[0,1].grid(True)
    
    if agent.losses:
        axes[1,0].plot(agent.losses)
        axes[1,0].set_title('Training Loss')
        axes[1,0].set_xlabel('Update Step')
        axes[1,0].set_ylabel('Loss')
        axes[1,0].grid(True)
    
    episodes = range(len(agent.rewards_history))
    epsilon_values = [max(0.05, 1.0 - i * 5e-6) for i in episodes]
    axes[1,1].plot(episodes, epsilon_values)
    axes[1,1].set_title('Epsilon Decay')
    axes[1,1].set_xlabel('Episode')
    axes[1,1].set_ylabel('Epsilon')
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    
    print(f"Training progress plot saved to {save_path}")

def detailed_evaluation(agent, env, num_episodes=20):
    print("\n=== DETAILED EVALUATION ===")
    
    success_count = 0
    total_rewards = []
    episode_lengths = []
    action_counts = {i: 0 for i in range(agent.action_dim)}
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            action = agent.select_action(state[0], state[1], training=False)
            action_counts[action] += 1
            
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if done:
                if episode_reward > 0:
                    success_count += 1
                break
            
            state = next_state
        
        total_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Eval Episode {episode+1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    success_rate = success_count / num_episodes
    avg_reward = np.mean(total_rewards)
    avg_length = np.mean(episode_lengths)
    
    print(f"\nEvaluation Results:")
    print(f"  Success Rate: {success_rate:.2%}")
    print(f"  Average Reward: {avg_reward:.2f}")
    print(f"  Average Episode Length: {avg_length:.1f}")
    print(f"  Action Distribution: {action_counts}")
    
    return {
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'avg_length': avg_length,
        'action_distribution': action_counts
    }

def decide_next_steps(stats, eval_results):
    print("\n=== DECISION FRAMEWORK ===")
    
    recommendations = []
    
    if eval_results['success_rate'] == 0:
        recommendations.append("CRITICAL: Success rate = 0%. Need to check reward function and environment!")
    elif eval_results['success_rate'] < 0.2:
        recommendations.append("SUCCESS RATE LOW: Need more training or hyperparameter adjustment")
    else:
        recommendations.append("SUCCESS RATE OK: Model is learning!")
    
    if stats['avg_reward'] < -10:
        recommendations.append("REWARD TOO LOW: Check reward function")
    elif stats['avg_reward'] < 0:
        recommendations.append("NEGATIVE REWARD: Might need more training")
    else:
        recommendations.append("REWARD POSITIVE: Good progress!")
    
    if stats['epsilon'] > 0.5:
        recommendations.append("EPSILON HIGH: Still exploring a lot, need more training")
    else:
        recommendations.append("EPSILON OK: Switched to exploitation")
    
    action_dist = eval_results['action_distribution']
    max_action_count = max(action_dist.values())
    total_actions = sum(action_dist.values())
    
    if max_action_count / total_actions > 0.8:
        recommendations.append("ACTION BIAS: Agent might be stuck with one action")
    else:
        recommendations.append("ACTION VARIETY: Agent uses diverse actions")
    
    return recommendations

def suggest_next_steps(recommendations, stats, eval_results):
    print("\n=== SUGGESTED NEXT STEPS ===")
    
    if eval_results['success_rate'] > 0.2 and stats['avg_reward'] > -5:
        print("SCENARIO: Model is learning well!")
        print("   → Continue training for 5000-10000 more episodes")
        print("   → Monitor for overfitting")
        print("   → Consider reducing learning rate")
        return "continue_training"
    
    elif eval_results['success_rate'] > 0 and stats['avg_reward'] > -20:
        print("SCENARIO: Model is learning slowly but making progress")
        print("   → Continue training for 10000-20000 more episodes")
        print("   → Consider increasing learning rate")
        print("   → Check if reward function needs adjustment")
        return "slow_progress"
    
    else:
        print("SCENARIO: Model is not learning!")
        print("   → STOP training, need debugging")
        print("   → Check reward function")
        print("   → Check environment")
        print("   → Might need architecture changes")
        return "debug_needed"

def post_training_analysis(agent, env):
    print("Starting post-training analysis...")
    
    stats = analyze_training_results(agent)
    
    plot_training_progress(agent)
    
    eval_results = detailed_evaluation(agent, env)
    
    recommendations = decide_next_steps(stats, eval_results)
    
    print("\n=== RECOMMENDATIONS ===")
    for rec in recommendations:
        print(f"  {rec}")
    
    next_step = suggest_next_steps(recommendations, stats, eval_results)
    
    return {
        'stats': stats,
        'eval_results': eval_results,
        'recommendations': recommendations,
        'next_step': next_step
    }

def quick_fixes():
    fixes = {
        'success_rate_0': """
        Fix 1: Success rate = 0%
        - Check if reward function is correct
        - Ensure episodes can terminate
        - Check if action space is reasonable
        """,
        
        'reward_too_low': """
        Fix 2: Reward too low
        - Increase positive reward for good actions
        - Reduce negative penalty
        - Add shaped reward
        """,
        
        'action_bias': """
        Fix 3: Agent stuck with one action
        - Increase epsilon for more exploration
        - Check action masking
        - Add penalty for repetitive actions
        """,
        
        'unstable_training': """
        Fix 4: Training unstable
        - Reduce learning rate
        - Increase batch size
        - Add gradient clipping
        """
    }
    
    return fixes

if __name__ == "__main__":
    print("Post-training analysis toolkit ready!")
    print("Run: post_training_analysis(agent, env)")