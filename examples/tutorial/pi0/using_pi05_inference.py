import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors

# Swap this import per-policy
from lerobot.policies.pi05 import PI05Policy
import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
os.environ['HF_HUB_OFFLINE'] = 'True'
# os.environ['HF_HUB_ONLINE'] = 'True'
# os.system("export all_proxy=http://183.173.54.227:7890/")
# os.system("export ALL_PROXY=http://183.173.54.227:7890/")

# load a policy
model_id = "/data2/wyn/models/lerobot/pi05_base"  # <- swap checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy = PI05Policy.from_pretrained(model_id).to(device).eval()

preprocess, postprocess = make_pre_post_processors(
    policy.config,
    model_id,
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)
# load a lerobotdataset (we will replace with a simpler dataset)
# dataset = LeRobotDataset("lerobot/libero")
dataset = LeRobotDataset("/data2/wyn/dataset/IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot/")
# # dataset = LeRobotDataset("/media/wyn/data/10-EmbodiedAI/dataset/svla_so101_pickplace")
# dataset = LeRobotDataset("/media/wyn/data/10-EmbodiedAI/dataset/GM-100/gm100-cobotmagic-lerobot/task_00006/")

# pick an episode
episode_index = 0

# each episode corresponds to a contiguous range of frame indices
from_idx = dataset.meta.episodes["dataset_from_index"][episode_index]
to_idx   = dataset.meta.episodes["dataset_to_index"][episode_index]

# get a single frame from that episode (e.g. the first frame)
frame_index = from_idx
frame = dict(dataset[frame_index])

batch = preprocess(frame)
with torch.inference_mode():
    pred_action = policy.select_action(batch)
    # use your policy postprocess, this post process the action
    # for instance unnormalize the actions, detokenize it etc..
    pred_action = postprocess(pred_action)


