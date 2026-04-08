from lerobot.datasets.lerobot_dataset import LeRobotDataset
from datasets import load_dataset
# dataset = LeRobotDataset("lerobot/libero")
# dataset = load_dataset("GayStarc/libero_object_no_noops_1.0.0_lerobot_v3.0")
dataset = load_dataset("GayStarc/libero_10_no_noops_1.0.0_lerobot_v3.0")
# sample = dataset[0]
# print(sample.keys())                # 查看顶层键
# obs = sample.get("observation", {})
# print(type(obs), obs.keys())        # 查看observation的结构
# dict_keys(['observation.images.image', 'observation.images.image2', 'observation.state', 'action', 'timestamp',
# 'frame_index', 'episode_index', 'index', 'task_index', 'task'])