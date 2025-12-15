from huggingface_hub import snapshot_download

repo_id = "facebook/s2t-small-librispeech-asr"
local_path = '/Users/leon/Library/Mobile Documents/com~apple~CloudDocs/Uni/Master/3.Semester/naoqi-docker/project/nao-says-robot-game/models'

local_dir = snapshot_download(repo_id, local_dir=local_path)
print("Heruntergeladen nach:", local_dir)