import os
import logging
from typing import Any, Dict

import torch
import torch.nn as nn

from utils.model import align_and_update_state_dicts

logger = logging.getLogger(__name__)


class BaseModel(nn.Module):
    def __init__(self, opt, module: nn.Module):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.model = module

    def forward(self, *inputs, **kwargs):
        outputs = self.model(*inputs, **kwargs)
        return outputs

    def save_pretrained(self, save_dir):
        state = self._get_tuned_state_dict()
        torch.save(state, os.path.join(save_dir, "model_state_dict.pt"))

    def from_pretrained(self, load_dir):
        state_dict = torch.load(load_dir, map_location=self.opt['device'])
        state_dict = align_and_update_state_dicts(
            self.model.state_dict(),
            state_dict,
            self.opt["SOLVER"].get('IGNORE_FIX', [])
        )
        self.model.load_state_dict(state_dict, strict=False)
        return self
    
    def _get_tuned_state_dict(self) -> Dict[str, Any]:
        state = self.model.state_dict()

        ignore_fix = self.opt["SOLVER"].get('IGNORE_FIX', [])

        if not ignore_fix:
            return state

        ret = {}

        for key, value in state.items():
            # Add only the keys that we care about
            for tuned_param_name in ignore_fix:
                if tuned_param_name in key:
                    ret[key] = value

        return ret