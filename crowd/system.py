"""
pairwise-crowd-system - CrowdSystem

This module contains a deliberately broken implementation.
Fix all bugs so that tests pass.
"""


class CrowdSystem:
    def __init__(self):
        self.state = {}

    def update_crowd(self, *args, **kwargs):
        """BUG: 人群更新只做简单路径跟随，不做避障"""
        # Intentionally broken implementation
        return None

    def calculate_avoidance(self, *args, **kwargs):
        """BUG: 避障只检查前方，不检查侧面和后方"""
        # Intentionally broken implementation
        return None

    def follow_path(self, *args, **kwargs):
        """BUG: 路径跟随只看下一个路径点，不看整个路径"""
        # Intentionally broken implementation
        return None

    def calculate_flocking(self, *args, **kwargs):
        """BUG: 群体行为只做聚合，不做分离和对齐"""
        # Intentionally broken implementation
        return None

    def render_crowd(self, *args, **kwargs):
        """BUG: 人群渲染不做实例化，每个人单独提交Draw Call"""
        # Intentionally broken implementation
        return None

