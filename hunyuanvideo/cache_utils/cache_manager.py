import torch
import copy
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
class cache_manager:
    def __init__(self, mode="slow", config_path=BASE_DIR/"xslim_hunyuan_strategy.pth"):
        self.enable_cache = True  
        self.replace_with_flash_attn = False

        self.num_steps = 50 
        self.curr_step = 0
        self.num_double_blocks = 20
        self.num_single_blocks = 40

        self.is_cache_step = False
        self.is_next_cache_step = False
        self.cached_patchified_index = None
        self.other_patchified_index = None

        self.skip_token_num_list = []
        self.skip_num_step_length = 0
        self.txt_seq_len = None
        self.img_seq_len = None

        self.drop_block_cnt = {"double": None, "single": None}

        self.thresh = {"double": 12, "single": 25}  
        self.start_block = {"double": 3, "single": 3}
        self.accumulated_score = 0

        self.skip_block_index = {"double": [], "single": []}
        self.pre_skip_block_index = {"double": [], "single": []}

        self.block_cache_data = {
            "double": {"img": {}, "txt": {}},
            "single": {},
        }
        self.block_cache_info = {"double": [], "single": []}

        self.sample_ratio = None
        self.fully_infer_step = []
        self.block_refresh_step = []
        self.token_refresh_step = []

        self.config_path = config_path
        self.set_mode(mode)

    def _load_config(self, mode):
        cfg_all = torch.load(self.config_path, map_location="cpu")
        assert mode in cfg_all, f"unknown mode: {mode}"
        cfg = cfg_all[mode]
        self.sample_ratio = float(cfg["sample_ratio"])
        self.fully_infer_step = cfg["fully_infer_step"]
        self.block_refresh_step = cfg["block_refresh_step"]
        self.token_refresh_step = cfg["token_refresh_step"]

    def _init_cache_config(self, latents):
        self.drop_cnt = torch.zeros((latents.shape[1]), device=latents.device) - self.num_steps
        self.drop_block_cnt["double"] = torch.zeros(self.num_double_blocks, device=latents.device)
        self.drop_block_cnt["single"] = torch.zeros(self.num_single_blocks, device=latents.device)
        self.seq_len = latents.shape[1]
        avg_skip_token_num = int((1 - self.sample_ratio) * self.seq_len)
        self.skip_token_num_list = [avg_skip_token_num for _ in range(self.num_steps)]
    
    def set_mode(self, mode):
        self.mode = mode
        self._load_config(mode)

    def reset_cache(self):
        if self.skip_block_index["double"] or self.skip_block_index["single"]:
            self.pre_skip_block_index = copy.deepcopy(self.skip_block_index)
        self.skip_block_index = {"double": [], "single": []}
        self.block_cache_data = {"double": {"img": {}, "txt": {}}, "single": {}}
        self.double_accumulated_score = 0
        self.single_accumulated_score = 0

    def blockcache_selection(self, index, outp, inp, mod):
        l1 = (outp - inp).abs().mean()
        self.accumulated_score += l1.item()
        if self.accumulated_score < self.thresh[mod] and index > self.start_block[mod]:
            self.skip_block_index[mod].append(index)
            self.drop_block_cnt[mod][index] += 1
            block_should_record = True
        else:
            block_should_record = False
            self.accumulated_score = 0
        return block_should_record
    
    def tokencache_selection(self, diff):
        metric = diff.mean(dim=-1).view(-1)
        current_skip_num = self.skip_token_num_list[self.curr_step]
        indices = torch.sort(metric, dim=0, descending=False).indices
        cached_patchified_indices = indices[:current_skip_num]
        other_patchified_indices = indices[current_skip_num:]
        return cached_patchified_indices, other_patchified_indices

    def check_cache_step(self):
        self.is_fully_infer_step = False
        self.is_block_refresh_step = False
        self.is_token_refresh_step = False
        self.is_cache_step = False
        self.is_cache_blockdata = False
        self.is_cache_tokendata = False

        self.next_fully_infer_step = next((n for n in self.fully_infer_step if n > self.curr_step), None)
        if self.curr_step in self.fully_infer_step:
            self.is_fully_infer_step = True
        elif self.curr_step in self.block_refresh_step:
            self.is_block_refresh_step = True
        elif self.curr_step in self.token_refresh_step:
            self.is_token_refresh_step = True
        else:
            self.is_cache_step = True

        if self.curr_step in self.fully_infer_step or self.curr_step in self.block_refresh_step or self.curr_step in self.token_refresh_step:
            self.next_block_refresh_step = next((n for n in self.block_refresh_step if self.curr_step < n < self.next_fully_infer_step), None)
            self.next_token_refresh_step = next((n for n in self.token_refresh_step if self.curr_step < n < self.next_fully_infer_step), None)
            self.is_cache_blockdata = bool(self.next_block_refresh_step)
            self.is_cache_tokendata = bool(self.next_token_refresh_step)


MANAGER = cache_manager() 
