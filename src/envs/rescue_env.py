import gym
import numpy as np
import cv2
from gym_minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper

class RescueEnv(gym.Env):
    # Create MiniGrid, tensor RGB and Bounding box the victim
    def __init__(self,
                 env_name: str = "MiniGrid-Empty-8x8-v0",
                 img_size: int = 224,
                 max_boxes: int = 5):
        super().__init__()
        base = gym.make(env_name)
        base = RGBImgObsWrapper(base)   
        base = ImgObsWrapper(base)     
        self.env = base

        self.img_size = img_size
        self.max_boxes = max_boxes

        grid_w = self.env.unwrapped.grid.width
        self.tile_size = img_size // grid_w

        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(
                low=0, high=255,
                shape=(3, img_size, img_size),
                dtype=np.uint8
            ),
            "boxes": gym.spaces.Box(
                low=0, high=img_size,
                shape=(self.max_boxes, 4),
                dtype=np.int32
            )
        })
        self.action_space = self.env.action_space

    def reset(self):
        obs = self.env.reset()
        return self._process(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        processed = self._process(obs)
        info["raw_obs"] = obs
        return processed, reward, done, info

    def render(self, mode="human"):
        return self.env.render(mode)

    def _process(self, obs):
        img = obs                   
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.transpose(2, 0, 1)           
        boxes = self._extract_boxes()

        if len(boxes) < self.max_boxes:
            pad = np.zeros((self.max_boxes - len(boxes), 4), dtype=np.int32)
            boxes = np.vstack([boxes, pad])
        else:
            boxes = boxes[: self.max_boxes]

        return {"image": img, "boxes": boxes}

    def _extract_boxes(self) -> np.ndarray:
        grid = self.env.unwrapped.grid
        boxes = []
        for j in range(grid.width):
            for i in range(grid.height):
                cell = grid.get(j, i)
                if cell and cell.type == "goal":
                    x1 = j * self.tile_size
                    y1 = i * self.tile_size
                    x2 = x1 + self.tile_size
                    y2 = y1 + self.tile_size
                    boxes.append([x1, y1, x2, y2])
        return np.array(boxes, dtype=np.int32)

