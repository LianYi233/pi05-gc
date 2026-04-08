# from: https://huggingface.co/lerobot/smolvla_base
# lerobot-train --dataset.repo_id=IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot --output_dir=./outputs/test1/ --job_name=test-10 --policy.repo_id=lerobot/smolvla_base --policy.path=/media/wyn/data/10-EmbodiedAI/models/lerobot/smolvla_base --policy.dtype=bfloat16 --policy.device=cuda --steps=100000 --batch_size=4
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors

# Swap this import per-policy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# load a policy
# model_id = "lerobot/smolvla_base"  # <- swap checkpoint
model_id = "/media/wyn/data/10-EmbodiedAI/models/lerobot/smolvla_base/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy = SmolVLAPolicy.from_pretrained(model_id).to(device).eval()

preprocess, postprocess = make_pre_post_processors(
    policy.config,
    model_id,
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)
# load a lerobotdataset
# dataset = LeRobotDataset("lerobot/libero")
# dataset = LeRobotDataset("/media/wyn/data/10-EmbodiedAI/dataset/IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot/")

dataset = LeRobotDataset("/media/wyn/data/10-EmbodiedAI/dataset/svla_so101_pickplace")
# dataset = LeRobotDataset("/media/wyn/data/10-EmbodiedAI/dataset/GM-100/gm100-cobotmagic-lerobot-v3.0/task_00006/")

# pick an episode
episode_index = 0

# each episode corresponds to a contiguous range of frame indices
from_idx = dataset.meta.episodes["dataset_from_index"][episode_index]
to_idx   = dataset.meta.episodes["dataset_to_index"][episode_index]

# get a single frame from that episode (e.g. the first frame)
# frame_index = from_idx
frame_index = int(from_idx)
frame = dict(dataset[frame_index])

batch = preprocess(frame)

with torch.inference_mode():
    # pred_action = policy.select_action(frame)
    pred_action = policy.select_action(batch)
    # use your policy postprocess, this post process the action
    # for instance unnormalize the actions, detokenize it etc..
    pred_action = postprocess(pred_action)


# Evaluate a policy on the LIBERO benchmark
# lerobot-eval --policy.path=/media/wyn/data/10-EmbodiedAI/models/lerobot/smolvla_base/ --env.type=libero --env.task=libero_object --eval.n_episodes=10