import math

from crowd.system import CrowdSystem


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class TestCrowdSystem:
    # --- update_crowd -------------------------------------------------

    def test_update_crowd_applies_avoidance(self):
        """update_crowd() runs avoidance so head-on agents do not overlap."""
        agents = [
            {"id": "a", "position": (-2.0, 0.0), "velocity": (3.0, 0.0), "radius": 0.5},
            {"id": "b", "position": (2.0, 0.0), "velocity": (-3.0, 0.0), "radius": 0.5},
        ]
        CrowdSystem().update_crowd(agents, dt=0.1)
        assert dist(agents[0]["position"], agents[1]["position"]) >= 1.0

    def test_update_crowd_density_control(self):
        """update_crowd() slows agents down when local density is high."""
        agents = [{"id": "center", "position": (0.0, 0.0),
                   "velocity": (4.0, 0.0), "radius": 0.3, "max_speed": 4.0}]
        for i in range(10):
            angle = i / 10.0 * 2 * math.pi
            agents.append({"id": f"n{i}", "position": (0.5 * math.cos(angle),
                                                        0.5 * math.sin(angle)),
                           "velocity": (0.0, 0.0), "radius": 0.3, "max_speed": 4.0})
        updates = CrowdSystem().update_crowd(agents, dt=1.0)
        center = next(u for u in updates if u["id"] == "center")
        assert center["density"] > CrowdSystem.COMFORTABLE_DENSITY
        assert center["density_scale"] < 1.0

    def test_update_crowd_individual_differences(self):
        """update_crowd() preserves per-agent speed and animation variation."""
        slow = {"id": "old", "position": (0.0, -10.0), "velocity": (0.0, 4.0),
                "max_speed": 1.0, "path": [(0.0, -10.0), (0.0, 10.0)]}
        fast = {"id": "young", "position": (0.0, -10.0), "velocity": (0.0, 4.0),
                "max_speed": 6.0, "path": [(0.0, -10.0), (0.0, 10.0)]}
        updates = CrowdSystem().update_crowd([slow, fast], dt=0.1)
        speeds = {u["id"]: u["speed"] for u in updates}
        phases = {u["id"]: u["anim_phase"] for u in updates}
        assert speeds["young"] > speeds["old"]
        assert phases["young"] != phases["old"]

    # --- calculate_avoidance ------------------------------------------

    def test_avoidance_checks_all_directions(self):
        """calculate_avoidance() reacts to threats from every direction."""
        system = CrowdSystem()
        for threat_pos in [(-1.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]:
            agent = {"id": "me", "position": (0.0, 0.0),
                     "velocity": (0.0, 0.0), "radius": 0.5}
            other = {"id": "other", "position": threat_pos,
                     "velocity": (0.0, 0.0), "radius": 0.5}
            result = system.calculate_avoidance(agent, [other])
            assert result["checked_sectors"] == 8
            assert len(result["threats"]) == 1
            away_from_other = (agent["position"][0] - threat_pos[0],
                               agent["position"][1] - threat_pos[1])
            dot = (result["force"][0] * away_from_other[0]
                   + result["force"][1] * away_from_other[1])
            assert dot > 0

    def test_avoidance_predicts_collisions(self):
        """calculate_avoidance() uses predicted future positions, not current."""
        agent = {"id": "me", "position": (0.0, 0.0),
                 "velocity": (0.0, 2.0), "radius": 0.5}
        other = {"id": "other", "position": (1.2, 3.0),
                 "velocity": (0.0, -2.0), "radius": 0.5}
        assert dist(agent["position"], other["position"]) > 1.5
        result = CrowdSystem().calculate_avoidance(agent, [other], lookahead=2.0)
        assert len(result["threats"]) == 1
        assert 0.0 < result["threats"][0]["time_to_collision"] <= 2.0

    def test_avoidance_respects_priority(self):
        """calculate_avoidance() makes low-priority agents yield more."""
        def force(priority):
            system = CrowdSystem()
            agent = {"id": "me", "position": (0.0, 0.0),
                     "velocity": (0.0, 2.0), "radius": 0.5, "priority": priority}
            other = {"id": "other", "position": (1.0, 2.0),
                     "velocity": (0.0, -2.0), "radius": 0.5, "priority": 2.0}
            return math.hypot(*system.calculate_avoidance(agent, [other])["force"])
        assert force(0.5) > force(5.0)

    # --- follow_path --------------------------------------------------

    def test_follow_path_uses_whole_path(self):
        """follow_path() aims at a lookahead point, not the next waypoint."""
        agent = {"id": "walker", "position": (0.0, 0.5),
                 "max_speed": 4.0}
        path = [(0.0, 0.0), (0.0, 2.0), (3.0, 2.0)]
        result = CrowdSystem().follow_path(agent, path, lookahead=3.0)
        lx, ly = result["lookahead_point"]
        assert ly == 2.0
        assert lx > 0.0

    def test_follow_path_smooths_turns(self):
        """follow_path() decelerates for sharp corners instead of zig-zagging."""
        agent = {"id": "walker", "position": (0.0, 4.0), "max_speed": 4.0}
        system = CrowdSystem()
        corner = system.follow_path(agent, [(0.0, 0.0), (0.0, 5.0), (5.0, 5.0)])
        straight = system.follow_path(agent, [(0.0, 0.0), (0.0, 10.0)])
        assert corner["turn_angle"] > math.radians(80)
        assert corner["target_speed"] < straight["target_speed"]

    def test_follow_path_considers_density(self):
        """follow_path() slows down when the path runs through a crowd."""
        agent = {"id": "walker", "position": (0.0, 0.0), "max_speed": 4.0}
        neighbors = [
            {"id": f"n{i}", "position": (0.3 * math.cos(a),
                                          0.5 + 0.3 * math.sin(a)),
             "velocity": (0.0, 0.0), "radius": 0.3}
            for i, a in enumerate(j / 8.0 * 2 * math.pi for j in range(8))
        ]
        dense = CrowdSystem().follow_path(agent, [(0.0, 0.0), (0.0, 5.0)], neighbors)
        open_path = CrowdSystem().follow_path(agent, [(0.0, 0.0), (0.0, 5.0)])
        assert dense["density"] > CrowdSystem.COMFORTABLE_DENSITY
        assert dense["target_speed"] < open_path["target_speed"]

    # --- calculate_flocking -------------------------------------------

    def test_flocking_has_separation_alignment_cohesion(self):
        """calculate_flocking() applies separation, alignment and cohesion."""
        agent = {"id": "me", "position": (0.0, 0.0),
                 "velocity": (0.0, 0.0)}
        neighbors = [
            {"id": "close", "position": (0.8, 0.0), "velocity": (1.0, 0.0)},
            {"id": "far", "position": (-3.0, 0.0), "velocity": (1.0, 0.0)},
        ]
        result = CrowdSystem().calculate_flocking(agent, neighbors)
        assert result["separation"][0] < 0
        assert result["alignment"] == (1.0, 0.0)
        assert result["cohesion"][0] < 0

    def test_flocking_leadership(self):
        """calculate_flocking() picks one leader and steers followers."""
        leader = {"id": "leader", "position": (0.0, 1.0),
                  "velocity": (2.0, 0.0), "leadership": 5.0}
        follower = {"id": "follower", "position": (0.0, 0.0),
                    "velocity": (0.0, 0.0), "leadership": 0.0}
        other = {"id": "other", "position": (0.0, 2.0),
                 "velocity": (0.0, 0.0), "leadership": 0.0}
        system = CrowdSystem()
        f_result = system.calculate_flocking(follower, [leader, other])
        l_result = system.calculate_flocking(leader, [follower, other])
        assert f_result["leader_id"] == "leader"
        assert f_result["leadership"][0] > 0
        assert l_result["is_leader"] is True

    def test_flocking_goal_directed(self):
        """calculate_flocking() steers the whole group toward a shared goal."""
        agent = {"id": "me", "position": (0.0, 0.0), "velocity": (0.0, 0.0)}
        with_goal = CrowdSystem().calculate_flocking(agent, [], goal=(10.0, 0.0))
        without_goal = CrowdSystem().calculate_flocking(agent, [], goal=None)
        assert with_goal["goal_force"][0] > 0
        assert with_goal["force"][0] > 0
        assert without_goal["goal_force"] == (0.0, 0.0)

    # --- render_crowd -------------------------------------------------

    def test_render_crowd_instancing_and_animation_reuse(self):
        """render_crowd() batches draw calls and shares animation clips."""
        agents = [
            {"id": f"p{i}", "position": (float(i), 0.0),
             "mesh": "villager", "anim_state": "walk"}
            for i in range(50)
        ]
        result = CrowdSystem().render_crowd(agents, camera=(0.0, 0.0))
        assert result["instanced"] is True
        assert result["draw_calls"] < result["agent_count"]
        assert result["animation_clips"] < result["agent_count"]

    def test_render_crowd_lod(self):
        """render_crowd() uses cheaper LODs and static poses far away."""
        agents = [
            {"id": "near", "position": (5.0, 0.0), "anim_state": "walk"},
            {"id": "mid", "position": (50.0, 0.0), "anim_state": "walk"},
            {"id": "far", "position": (200.0, 0.0), "anim_state": "walk"},
        ]
        result = CrowdSystem().render_crowd(agents, camera=(0.0, 0.0))
        lods = result["lod_assignment"]
        assert lods["near"] == "high"
        assert lods["mid"] == "medium"
        assert lods["far"] == "low"
        low_batch = next(b for b in result["batches"] if b["lod"] == "low")
        assert low_batch["clip"]["frames"] == 1
