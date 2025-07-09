import os
import numpy as np
import cv2
from src.envs.rescue_env import RescueEnv

def main():
    env = RescueEnv(
        env_name="MiniGrid-Empty-8x8-v0",
        img_size=224,
        max_boxes=5
    )

    obs = env.reset()
    img = obs["image"]  
    img = np.transpose(img, (1, 2, 0))  

    os.makedirs("snapshots", exist_ok=True)
    cv2.imwrite("snapshots/snapshot.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    np.savetxt("snapshots/boxes.txt", obs["boxes"], fmt='%d')

    print("Saved: snapshots/snapshot.png")
    print("Saved: snapshots/boxes.txt")

if __name__ == "__main__":
    main()
