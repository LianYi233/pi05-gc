from huggingface_hub import snapshot_download
import os
# os.system("export http_proxy=http://183.173.54.227:7890/")
# os.system("export https_proxy=http://183.173.54.227:7890/")
# os.system("export HTTP_PROXY=http://183.173.54.227:7890/")
# os.system("export all_proxy=http://183.173.54.227:7890/")
# os.system("export ALL_PROXY=http://183.173.54.227:7890/")
snapshot_download(repo_id="lerobot/libero", repo_type="dataset", local_dir='/media/wyn/data/10-EmbodiedAI/dataset/')