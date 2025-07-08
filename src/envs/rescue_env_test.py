from rescue_env import RescueEnv
env = RescueEnv()
o = env.reset()
print(o["image"].shape, o["boxes"].shape)
