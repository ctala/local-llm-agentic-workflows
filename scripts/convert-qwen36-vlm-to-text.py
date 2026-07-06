#!/usr/bin/env python3
import json, os, shutil
from pathlib import Path
from safetensors.torch import load_file, save_file

src = Path('~/vllm/qwen3.6-35b-a3b-bf16')
dst = Path('~/vllm/qwen3.6-35b-a3b-bf16-textonly')
dst.mkdir(parents=True, exist_ok=True)

config = json.load(open(src / 'config.json'))
text_cfg = config['text_config']
text_cfg['architectures'] = ['Qwen3_5MoeForCausalLM']
# Remove MTP fields not present in text-only causal LM
text_cfg.pop('mtp_num_hidden_layers', None)
text_cfg.pop('mtp_use_dedicated_embeddings', None)
json.dump(text_cfg, open(dst / 'config.json', 'w'), indent=2)

# Copy tokenizer files
for fname in ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
              'added_tokens.json', 'preprocessor_config.json']:
    fpath = src / fname
    if fpath.exists():
        shutil.copy2(fpath, dst / fname)

idx = json.load(open(src / 'model.safetensors.index.json'))
new_weight_map = {}
for old_name, shard in idx['weight_map'].items():
    if old_name.startswith('model.language_model.'):
        new_name = old_name.replace('model.language_model.', 'model.', 1)
        new_weight_map[new_name] = shard
    elif old_name.startswith('lm_head.'):
        new_weight_map[old_name] = shard
    else:
        # skip mtp.* and vision weights
        pass

# Map old shard -> new keys
shard_to_keys = {}
for new_name, shard in new_weight_map.items():
    shard_to_keys.setdefault(shard, []).append(new_name)

old_shard_files = set(new_weight_map.values())
for shard in sorted(old_shard_files):
    print(f'Processing {shard} ...')
    tensors = load_file(src / shard)
    new_tensors = {}
    for old_name, tensor in tensors.items():
        if old_name.startswith('model.language_model.'):
            new_name = old_name.replace('model.language_model.', 'model.', 1)
            new_tensors[new_name] = tensor
        elif old_name.startswith('lm_head.'):
            new_tensors[old_name] = tensor
    if not new_tensors:
        continue
    save_file(new_tensors, dst / shard, metadata={'format': 'pt'})

# Build new index preserving original shard grouping
new_index = {'metadata': idx.get('metadata', {}), 'weight_map': {}}
for new_name, shard in new_weight_map.items():
    new_index['weight_map'][new_name] = shard

# Remove any shards that became empty from the index (if any)
used_shards = set(new_index['weight_map'].values())
json.dump(new_index, open(dst / 'model.safetensors.index.json', 'w'), indent=2)

# Remove empty shards
for p in dst.glob('model-*.safetensors'):
    if p.name not in used_shards:
        p.unlink()

print('Done. Output:', dst)
print('Keys:', len(new_index['weight_map']))
