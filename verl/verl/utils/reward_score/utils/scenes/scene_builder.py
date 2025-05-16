from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Union, Optional
from functools import cached_property

import torch
import numpy as np
from gymnasium import spaces
import sapien
from transforms3d.euler import euler2quat

if TYPE_CHECKING:
    from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs import Actor, Articulation
from mani_skill.utils.structs.types import Array
from mani_skill.utils.structs.pose import Pose
from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.envs.utils.randomization.pose import random_quaternions



class SceneBuilder:
    """Base class for defining scene builders that can be reused across tasks"""

    env: BaseEnv
    """Env which scenebuilder will build in."""

    robot_init_qpos_noise: float = 0.02
    """Robot init qpos noise"""

    builds_lighting: bool = False
    """Whether this scene builder will add its own lighting when build is called. If False, ManiSkill will add some default lighting"""

    build_configs: Optional[List[Any]] = None
    """List of scene configuration information that can be used to **build** scenes during reconfiguration (i.e. `env.reset(seed=seed, options=dict(reconfigure=True))`). Can be a dictionary, a path to a json file, or some other data. If a scene needs to load build config data, it will index/sample such build configs from this list."""
    init_configs: Optional[List[Any]] = None
    """List of scene configuration information that can be used to **init** scenes during reconfiguration (i.e. `env.reset()`). Can be a dictionary, a path to a json file, or some other data. If a scene needs to load init config data, it will index/sample such init configs from this list."""

    scene_objects: Optional[Dict[str, Actor]] = None
    """Scene objects are any dynamic, kinematic, or static Actor built by the scene builder. Useful for accessing objects in the scene directly."""
    movable_objects: Optional[Dict[str, Actor]] = None
    """Movable objects are any **dynamic** Actor built by the scene builder. movable_objects is a subset of scene_objects. Can be used to query dynamic objects for e.g. task initialization."""
    articulations: Optional[Dict[str, Articulation]] = None
    """Articulations are any articulation loaded in by the scene builder."""

    navigable_positions: Optional[List[Union[Array, spaces.Box]]] = None
    """Some scenes allow for mobile robots to move through these scene. In this case, a list of navigable positions per env_idx (e.g. loaded from a navmesh) should be provided for easy initialization. Can be a discretized list, range, spaces.Box, etc."""

    def __init__(self, env, robot_init_qpos_noise=0.02):
        self.env = env
        self.robot_init_qpos_noise = robot_init_qpos_noise

    def build(self, build_config_idxs: List[int] = None):
        """
        Should create actor/articulation builders and only build objects into the scene without initializing pose, qpos, velocities etc.
        """
        raise NotImplementedError()

    def initialize(self, env_idx: torch.Tensor, init_config_idxs: List[int] = None):
        """
        Should initialize the scene, which can include e.g. setting the pose of all objects, changing the qpos/pose of articulations/robots etc.
        """
        raise NotImplementedError()

    def sample_build_config_idxs(self) -> List[int]:
        """
        Sample idxs of build configs for easy scene randomization. Should be changed to fit shape of self.build_configs.
        """
        return torch.randint(
            low=0, high=len(self.build_configs), size=(self.env.num_envs,)
        ).tolist()

    def sample_init_config_idxs(self) -> List[int]:
        """
        Sample idxs of init configs for easy scene randomization. Should be changed to fit shape of self.init_configs.
        """
        return torch.randint(
            low=0, high=len(self.init_configs), size=(self.env.num_envs,)
        ).tolist()

    @cached_property
    def build_config_names_to_idxs(self) -> Dict[str, int]:
        return dict((v, i) for i, v in enumerate(self.build_configs))

    @cached_property
    def init_config_names_to_idxs(self) -> Dict[str, int]:
        return dict((v, i) for i, v in enumerate(self.init_configs))

    @property
    def scene(self):
        return self.env.scene


class RFSceneBuilder(SceneBuilder):
    def __init__(self, env, cfg, **kwargs):
        super().__init__(env, **kwargs)
        self.cfg = cfg
        self.first_initial = True
        self.env.annotation_data = {}

    def initialize(self, env_idx: torch.Tensor, collision_detect=True):
        if self.first_initial:
            self.first_initial = False
            print("First time initializing the scene, skip.")
            return
        b = len(env_idx)
        scene_cfg = self.cfg['scene']

        # primitive
        if 'primitives' in scene_cfg:
            for primitive_cfg in scene_cfg['primitives']:
                asset = getattr(self.env, primitive_cfg['name'], None)
                if not asset:
                    raise AttributeError(f'Attribute "{primitive_cfg["name"]}" not found in SceneBuilder.')
                qpos = np.array(primitive_cfg['pos']['qpos'])
                if 'randq_scale' in primitive_cfg:
                    qpos = np.array(qpos) + np.array(primitive_cfg['pos']['randq_scale'])* np.random.rand((len(qpos)))
                if 'random_quaternions' in primitive_cfg['pos']:
                    qpos = random_quaternions(
                        b,
                        lock_x=primitive_cfg['pos']['random_quaternions'][0],
                        lock_y=primitive_cfg['pos']['random_quaternions'][1],
                        lock_z=primitive_cfg['pos']['random_quaternions'][2]
                    )
                asset.set_pose(Pose.create_from_pq(primitive_cfg['pos']['ppos']['p'], qpos))

        # agents
        agent_pposes = []    # add collision avoidance
        if 'agents' in self.cfg:
            agents_cfg = self.cfg['agents']
            is_multi_agent = (len(agents_cfg) > 1)
            agent = self.env.agent
            self.articulations = {}
            for idx, agent_cfg in enumerate(agents_cfg):
                pos_cfg = agent_cfg['pos']
                ppos = pos_cfg['ppos']['p']
                temp_ppos = np.array(ppos)
                print(temp_ppos)
                if 'randp_scale' in pos_cfg:
                    available_pos = False
                    count = 0
                    while not available_pos:
                        if count > 5000:
                            print(f'Fail to find suitable position for {agent_cfg["robot_uid"]} in {count} time. Skip.')
                            exit(0)
                        available_pos = True
                        temp_ppos = np.array(ppos) + np.array(agent_cfg['pos']['randp_scale']) * np.random.rand((len(ppos)))
                        temp_ppos = temp_ppos.tolist()
                        for agent_ppos in agent_pposes:
                            delta_pos = np.abs(np.array(agent_ppos) - np.array(temp_ppos))
                            if np.max(delta_pos[:2]) < 0.6:
                                available_pos = False
                                if (agent_cfg['robot_uid'].startswith('panda')):
                                    if delta_pos[0] < 0.3:
                                        available_pos = False
                                    else:
                                        available_pos = True
                        if 'pseduo_assets_areas' in self.cfg:
                            for pseudo_asset_pos in self.cfg['pseduo_assets_areas']:
                                axis_min = pseudo_asset_pos['pos']['min']
                                axis_max = pseudo_asset_pos['pos']['max']
                                delta_min = np.array(temp_ppos) - np.array(axis_min)
                                delta_max = np.array(axis_max) - np.array(temp_ppos)
                                if delta_max.min() >= 0 and delta_min.min() >=0:    # overlap with the area
                                    available_pos = False
                        count += 1
                        if not collision_detect:
                            break
                euler_pose = pos_cfg['ppos']['q']
                if 'rand_euler' in pos_cfg:
                    new_euler_pose = np.array(euler_pose) + np.random.rand((len(euler_pose))) * np.array(agent_cfg['pos']['rand_euler'])
                    euler_pose = new_euler_pose.tolist()
                temp_ppos = sapien.Pose(temp_ppos, q=euler2quat(*euler_pose))
                agent_pposes.append(temp_ppos.p)
                qpos = np.array((pos_cfg['qpos']))
                if 'randq_scale' in pos_cfg:
                    qpos = np.tile(qpos, (b, 1)) + np.tile(np.array(agent_cfg['pos']['randq_scale']), (b, 1)) * np.random.rand(b, (len(qpos)))
                if 'random_quaternions' in agent_cfg['pos']:
                    qpos = random_quaternions(
                        b,
                        lock_x=agent_cfg['pos']['random_quaternions'][0],
                        lock_y=agent_cfg['pos']['random_quaternions'][1],
                        lock_z=agent_cfg['pos']['random_quaternions'][2]
                    )
                if is_multi_agent:
                    agent.agents[idx].reset(qpos)
                    agent.agents[idx].robot.set_pose(temp_ppos)
                    self.articulations[agent_cfg['robot_uid']] = agent.agents[idx]
                else:
                    agent.reset(qpos)
                    agent.robot.set_pose(temp_ppos)
                    self.articulations[agent_cfg['robot_uid']] = agent

            
        asset_pposes = []    # add collision avoidance
        asset_pposes.extend(agent_pposes)    # add agent to collision avoidance
        # objects
        if 'objects' in self.cfg:
            objects_cfg = self.cfg['objects']
            self.movable_objects = {}
            for asset_cfg in objects_cfg:
                asset = getattr(self.env, asset_cfg['name'], None)
                if not asset:
                    raise AttributeError(f'Attribute "{asset_cfg["name"]}" not found in SceneBuilder.')
                ppos = asset_cfg['pos']['ppos']['p']
                temp_ppos = np.array(ppos)
                if 'randp_scale' in asset_cfg['pos']:
                    available_pos = False
                    count = 0
                    while not available_pos:
                        if count > 100:
                            print(f'Fail to find suitable position for {asset_cfg["name"]} in {count} time. Skip.')
                            exit(0)
                        available_pos = True
                        temp_ppos = np.array(ppos) + np.array(asset_cfg['pos']['randp_scale']) * np.random.rand((len(ppos)))
                        temp_ppos = temp_ppos.tolist()
                        for object_ppos in asset_pposes:
                            delta_pos = np.abs(np.array(object_ppos) - np.array(temp_ppos))
                            if np.max(delta_pos[:2]) < 0.15:
                                available_pos = False
                        if 'pseduo_assets_areas' in self.cfg:
                            for pseudo_asset_pos in self.cfg['pseduo_assets_areas']:
                                axis_min = pseudo_asset_pos['pos']['min']
                                axis_max = pseudo_asset_pos['pos']['max']
                                delta_min = np.array(temp_ppos) - np.array(axis_min)
                                delta_max = np.array(axis_max) - np.array(temp_ppos)
                                if delta_max.min() >= 0 and delta_min.min() >=0:    # overlap with the area
                                    available_pos = False
                        count += 1
                        if not collision_detect:
                            break
                asset_pposes.append(temp_ppos)
                qpos = asset_cfg['pos']['qpos']
                if 'randq_scale' in asset_cfg['pos']:
                    qpos = np.array(qpos) + np.array(asset_cfg['pos']['randq_scale'])* np.random.rand((len(qpos)))
                if 'random_quaternions' in asset_cfg['pos']:
                    qpos = random_quaternions(
                        b,
                        lock_x=asset_cfg['pos']['random_quaternions'][0],
                        lock_y=asset_cfg['pos']['random_quaternions'][1],
                        lock_z=asset_cfg['pos']['random_quaternions'][2]
                    )
                asset.set_pose(Pose.create_from_pq(temp_ppos, qpos))
                self.movable_objects[asset_cfg['name']] = asset
        property_poses = self.get_property_poses()
        print('-'*10 + 'Property Poses' + '-'*10)
        for n, pose in property_poses.items():
            print(f'{n}: {pose}')

    def get_property_poses(self):
        property_poses = {}
        if getattr(self, 'scene_objects'):
            for n, obj in self.scene_objects.items():
                property_poses[n] = obj.pose
        if getattr(self, 'movable_objects'):
            for n, obj in self.movable_objects.items():
                property_poses[n] = obj.pose
        if getattr(self, 'articulations'):
            for n, obj in self.articulations.items():
                property_poses[n] = obj.robot.pose
        return property_poses