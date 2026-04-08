import os, torch
from accelerate import Accelerator

acc = Accelerator()
print("RANK", acc.process_index, "LOCAL_RANK", acc.local_process_index)
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("device_count =", torch.cuda.device_count())
print("current_device =", torch.cuda.current_device())
print("device_name =", torch.cuda.get_device_name(torch.cuda.current_device()))
