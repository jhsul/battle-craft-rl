import torch as th
import torch.nn as nn
import sys
import numpy as np

sys.path.insert(0, "vpt")  # nopep8

from gym3.types import DictType
from typing import Dict, Optional

from lib.action_mapping import CameraHierarchicalMapping
from lib.actions import ActionTransformer
from lib.policy import MinecraftAgentPolicy
from lib.scaled_mse_head import ScaledMSEHead

from lib.torch_util import default_device_type, set_default_torch_device
from vpt.agent import ACTION_TRANSFORMER_KWARGS, POLICY_KWARGS, PI_HEAD_KWARGS

class ConnectionNetwork(nn.Module):
    def __init__(self, latent_size):
        super().__init__()

        self.connection_layer1 = nn.Linear(latent_size,
                                          latent_size)
        self.connection_layer2 = nn.Linear(latent_size,
                                          latent_size)
        
    def forward(self, latent):
        latent = th.relu(self.connection_layer1(latent))
        latent = th.relu(self.connection_layer2(latent))
        return latent

class EfficientVPT(nn.Module):

    def __init__(self, env, device=None, policy_kwargs=None, pi_head_kwargs=None):

        if device is None:
            device = 'cpu'
        self.device = th.device(device)
        # Set the default torch device for underlying code as well
        set_default_torch_device(self.device)
        self.action_mapper = CameraHierarchicalMapping(n_camera_bins=11)
        action_space = self.action_mapper.get_action_space_update()
        action_space = DictType(**action_space)

        self.action_transformer = ActionTransformer(
            **ACTION_TRANSFORMER_KWARGS)

        if policy_kwargs is None:
            policy_kwargs = POLICY_KWARGS
        if pi_head_kwargs is None:
            pi_head_kwargs = PI_HEAD_KWARGS

        agent_kwargs = dict(policy_kwargs=policy_kwargs,
                            pi_head_kwargs=pi_head_kwargs, action_space=action_space)

        self.policy = MinecraftAgentPolicy(**agent_kwargs).to(device)

        # Just a "continuation" layer between them
        # TODO Use a residual connection here?
        self.connection_net = ConnectionNetwork(self.policy.net.output_latent_size())

        def make_value_head(v_out_size: int, norm_type: str = "ewma", norm_kwargs: Optional[Dict] = None):
            return ScaledMSEHead(v_out_size, 1, norm_type=norm_type, norm_kwargs=norm_kwargs)
        
        self.value_head = make_value_head(self.policy.net.output_latent_size())
        self.value_processor = ConnectionNetwork(self.policy.net.output_latent_size())
        
    def run_vpt_base(self, agent_obs, hidden_state, dummy_first):
        (latent, _), hidden_state_out = self.policy.net(
            agent_obs, hidden_state, t={"first": dummy_first})
        
        return latent, hidden_state_out

    def get_real_value(self, latent):
        latent = self.value_processor(latent)
        return self.value_head(latent)
    
    def get_policy(self, latent):
        latent = self.connection_net(latent)
        return self.policy.pi_head(latent)
    
    def get_aux_value(self, latent):
        latent = self.connection_net(latent)
        return self.policy.value_head(latent)
    
    def value_parameters(self):
        return list[self.value_head.parameters()] + list[self.value_processor.parameters()]
    
    def policy_parameters(self):
        return list[self.policy.pi_head.parameters()] + list[self.connection_net.parameters()]
    
    def aux_parameters(self):
        return list[self.policy.value_head.parameters()] + list[self.connection_net.parameters()]

    def load_vpt_weights(self, path):
        """
        Load model weights from a path
        Leaves the weights for non-VPT things alone
        """
        self.policy.load_state_dict(
            th.load(path, map_location=self.device), strict=False)
        # copy the weights from the vpt value head to the real value head as well
        self.value_head.load_state_dict(self.policy.value_head.state_dict())

    def load_weights(self, path):
        """
        Loads in weights native to this model type for baseline running
        """
        self.load_state_dict(
            th.load(path, map_location=self.device), strict=False)
