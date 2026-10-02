import pytest
from crowd.system import CrowdSystem


class TestCrowdSystem:

    def test_update_crowd(self):
        """人群更新只做简单路径跟随，不做避障"""
        obj = CrowdSystem()
        result = obj.update_crowd()
        # The broken implementation returns None; fixed version should return proper value
        self.assertIsNotNone(result)

    def test_calculate_avoidance(self):
        """避障只检查前方，不检查侧面和后方"""
        obj = CrowdSystem()
        result = obj.calculate_avoidance()
        # The broken implementation returns None; fixed version should return proper value
        self.assertIsNotNone(result)

    def test_follow_path(self):
        """路径跟随只看下一个路径点，不看整个路径"""
        obj = CrowdSystem()
        result = obj.follow_path()
        # The broken implementation returns None; fixed version should return proper value
        self.assertIsNotNone(result)

    def test_calculate_flocking(self):
        """群体行为只做聚合，不做分离和对齐"""
        obj = CrowdSystem()
        result = obj.calculate_flocking()
        # The broken implementation returns None; fixed version should return proper value
        self.assertIsNotNone(result)

    def test_render_crowd(self):
        """人群渲染不做实例化，每个人单独提交Draw Call"""
        obj = CrowdSystem()
        result = obj.render_crowd()
        # The broken implementation returns None; fixed version should return proper value
        self.assertIsNotNone(result)

