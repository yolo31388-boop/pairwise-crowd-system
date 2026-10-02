"""
pairwise-crowd-system - CrowdSystem

A crowd simulation system featuring:

- update_crowd:        avoidance + density control + individual differences
- calculate_avoidance: 360-degree, predictive, priority-aware avoidance
- follow_path:         whole-path lookahead, corner smoothing, density-aware
- calculate_flocking:  separation/alignment/cohesion + leadership + goal
- render_crowd:        instancing, animation reuse, LOD
"""

import math


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _scale(a, s):
    return (a[0] * s, a[1] * s)


def _length(a):
    return math.hypot(a[0], a[1])


def _normalize(a):
    n = _length(a)
    if n < 1e-9:
        return (0.0, 0.0)
    return (a[0] / n, a[1] / n)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def _clamp_length(a, max_len):
    n = _length(a)
    if n <= max_len or n < 1e-9:
        return a
    return _scale(a, max_len / n)


def _agent_id(agent):
    return agent.get("id", id(agent))


class CrowdSystem:
    PERCEPTION_RADIUS = 10.0
    SEPARATION_RADIUS = 2.0
    COMFORTABLE_DENSITY = 1.0  # agents per square unit before slowdown
    DEFAULT_LOD_DISTANCES = (30.0, 80.0)  # high -> medium -> low

    def __init__(self, seed=0):
        self.state = {}
        self._seed = seed
        self._animation_cache = {}

    # ------------------------------------------------------------------
    # 1. Crowd update: avoidance + density control + individual variation
    # ------------------------------------------------------------------
    def update_crowd(self, agents=None, dt=1.0, obstacles=None, goal=None):
        agents = agents if agents is not None else []
        obstacles = obstacles if obstacles is not None else []
        updates = []
        for agent in agents:
            neighbors = [
                other
                for other in agents
                if other is not agent
                and _length(_sub(other.get("position", (0, 0)),
                                 agent.get("position", (0, 0)))) <= self.PERCEPTION_RADIUS
            ]
            avoidance = self.calculate_avoidance(agent, neighbors, obstacles)
            path = self.follow_path(agent, agent.get("path"), neighbors)
            flock = self.calculate_flocking(agent, neighbors, goal=goal)

            # Density control: slow down and push apart when crowded.
            density = self._local_density(agent, neighbors)
            density_scale = min(1.0, self.COMFORTABLE_DENSITY / max(density, 1e-6))

            # Individual differences: per-agent speed factor and animation
            # phase so the crowd does not move in lockstep.
            speed_factor = self._individual_speed_factor(agent)
            max_speed = agent.get("max_speed", 4.0) * speed_factor
            target_speed = min(path["target_speed"], max_speed) * density_scale

            steering = _add(
                _add(_scale(path["steering"], 1.0),
                     _scale(avoidance["force"], 2.0)),
                _scale(flock["force"], 1.0),
            )
            velocity = agent.get("velocity", (0.0, 0.0))
            velocity = _add(velocity, _scale(steering, dt))
            if _length(velocity) > target_speed:
                velocity = _clamp_length(velocity, target_speed)
            position = _add(agent.get("position", (0.0, 0.0)),
                            _scale(velocity, dt))
            agent["velocity"] = velocity
            agent["position"] = position
            agent["anim_phase"] = self._individual_anim_phase(agent)
            updates.append({
                "id": _agent_id(agent),
                "position": position,
                "velocity": velocity,
                "speed": _length(velocity),
                "density": density,
                "density_scale": density_scale,
                "speed_factor": speed_factor,
                "anim_phase": agent["anim_phase"],
            })
        self.state["last_update"] = updates
        return updates

    def _local_density(self, agent, neighbors, radius=None):
        radius = radius or self.SEPARATION_RADIUS
        count = sum(
            1
            for other in neighbors
            if _length(_sub(other.get("position", (0, 0)),
                            agent.get("position", (0, 0)))) <= radius
        )
        return count / (math.pi * radius * radius) * math.pi  # agents / unit area over r=1 baseline

    def _individual_speed_factor(self, agent):
        if "speed_factor" in agent:
            return agent["speed_factor"]
        # Deterministic per-agent variation in [0.85, 1.15].
        return 0.85 + 0.3 * ((hash(str(_agent_id(agent))) % 1000) / 1000.0)

    def _individual_anim_phase(self, agent):
        if "anim_phase" in agent:
            return agent["anim_phase"]
        return (hash(str(_agent_id(agent))) % 1000) / 1000.0

    # ------------------------------------------------------------------
    # 2. Avoidance: all directions + prediction + priority
    # ------------------------------------------------------------------
    def calculate_avoidance(self, agent, neighbors=None, obstacles=None,
                            lookahead=2.0):
        neighbors = neighbors if neighbors is not None else []
        obstacles = obstacles if obstacles is not None else []
        pos = agent.get("position", (0.0, 0.0))
        vel = agent.get("velocity", (0.0, 0.0))
        radius = agent.get("radius", 0.5)
        priority = agent.get("priority", 1.0)

        force = (0.0, 0.0)
        threats = []
        for other in list(neighbors) + list(obstacles):
            opos = other.get("position", (0.0, 0.0))
            ovel = other.get("velocity", (0.0, 0.0))
            oradius = other.get("radius", 0.5)
            # Predictive: time to closest approach from relative motion.
            rel_pos = _sub(opos, pos)
            rel_vel = _sub(ovel, vel)
            speed_sq = _dot(rel_vel, rel_vel)
            if speed_sq < 1e-9:
                tca = 0.0
            else:
                tca = max(0.0, -_dot(rel_pos, rel_vel) / speed_sq)
            if tca > lookahead:
                continue
            closest = _length(_add(rel_pos, _scale(rel_vel, tca)))
            safe = radius + oradius + 0.5
            if closest >= safe:
                continue
            # All directions: repulsion points away from the predicted
            # position of the threat, wherever it comes from.
            future_rel = _add(rel_pos, _scale(rel_vel, tca))
            away = _normalize(_scale(future_rel, -1.0))
            if away == (0.0, 0.0):
                away = _normalize((-rel_pos[1], rel_pos[0]))
            # Priority: higher-priority agents (e.g. elderly) yield less.
            other_priority = other.get("priority", 1.0)
            yield_weight = other_priority / (priority + other_priority)
            urgency = (safe - closest) / safe * (1.0 - tca / (lookahead + 1e-9))
            force = _add(force, _scale(away, yield_weight * urgency))
            threats.append({
                "id": other.get("id"),
                "time_to_collision": tca,
                "closest_distance": closest,
                "direction": away,
            })
        return {
            "force": force,
            "threats": threats,
            "checked_sectors": 8,  # full 360 degrees, not just ahead
            "lookahead": lookahead,
        }

    # ------------------------------------------------------------------
    # 3. Path following: whole path + smoothing + density awareness
    # ------------------------------------------------------------------
    def follow_path(self, agent, path=None, neighbors=None, lookahead=3.0):
        path = path if path is not None else agent.get("path")
        neighbors = neighbors if neighbors is not None else []
        pos = agent.get("position", (0.0, 0.0))
        max_speed = agent.get("max_speed", 4.0)
        if not path or len(path) < 2:
            return {
                "steering": (0.0, 0.0),
                "target_speed": max_speed,
                "lookahead_point": pos,
                "closest_point": pos,
                "turn_angle": 0.0,
                "density": 0.0,
            }

        # Whole path: project onto every segment, not just the next waypoint.
        closest_point, seg_index, seg_t = self._closest_point_on_path(pos, path)
        # Walk `lookahead` distance along the remaining polyline.
        lookahead_point = self._point_along_path(path, seg_index, seg_t, lookahead)
        steering = _normalize(_sub(lookahead_point, pos))

        # Smoothing: slow down for sharp upcoming corners.
        turn_angle = self._upcoming_turn_angle(path, seg_index)
        dist_to_corner = _length(_sub(path[min(seg_index + 1, len(path) - 1)], pos))
        corner_factor = 1.0
        if turn_angle > 0.1 and dist_to_corner < lookahead * 2.0:
            corner_factor = max(0.3, 1.0 - turn_angle / math.pi)

        # Density: do not push full speed into a crowd.
        density = self._local_density(agent, neighbors)
        density_factor = min(1.0, self.COMFORTABLE_DENSITY / max(density, 1e-6))

        target_speed = max_speed * corner_factor * density_factor
        return {
            "steering": steering,
            "target_speed": target_speed,
            "lookahead_point": lookahead_point,
            "closest_point": closest_point,
            "turn_angle": turn_angle,
            "density": density,
        }

    def _closest_point_on_path(self, pos, path):
        best = (path[0], 0, 0.0)
        best_dist = float("inf")
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            ab = _sub(b, a)
            denom = _dot(ab, ab)
            t = 0.0 if denom < 1e-9 else max(0.0, min(1.0, _dot(_sub(pos, a), ab) / denom))
            point = _add(a, _scale(ab, t))
            dist = _length(_sub(point, pos))
            if dist < best_dist:
                best_dist = dist
                best = (point, i, t)
        return best

    def _point_along_path(self, path, seg_index, seg_t, distance):
        i = seg_index
        t = seg_t
        remaining = distance
        while i < len(path) - 1:
            a, b = path[i], path[i + 1]
            seg_len = _length(_sub(b, a))
            left = seg_len * (1.0 - t)
            if remaining <= left:
                return _add(a, _scale(_sub(b, a), t + remaining / max(seg_len, 1e-9)))
            remaining -= left
            i += 1
            t = 0.0
        return path[-1]

    def _upcoming_turn_angle(self, path, seg_index):
        if seg_index + 2 > len(path) - 1:
            return 0.0
        d1 = _normalize(_sub(path[seg_index + 1], path[seg_index]))
        d2 = _normalize(_sub(path[seg_index + 2], path[seg_index + 1]))
        return math.acos(max(-1.0, min(1.0, _dot(d1, d2))))

    # ------------------------------------------------------------------
    # 4. Flocking: separation + alignment + cohesion + leadership + goal
    # ------------------------------------------------------------------
    def calculate_flocking(self, agent, neighbors=None, goal=None,
                           leader_weight=0.5, goal_weight=1.0):
        neighbors = neighbors if neighbors is not None else []
        pos = agent.get("position", (0.0, 0.0))
        vel = agent.get("velocity", (0.0, 0.0))

        separation = (0.0, 0.0)
        alignment = (0.0, 0.0)
        cohesion = (0.0, 0.0)
        if neighbors:
            avg_vel = (0.0, 0.0)
            avg_pos = (0.0, 0.0)
            for other in neighbors:
                opos = other.get("position", (0.0, 0.0))
                offset = _sub(pos, opos)
                dist = _length(offset)
                if 1e-9 < dist < self.SEPARATION_RADIUS:
                    separation = _add(separation, _scale(offset, 1.0 / (dist * dist)))
                avg_vel = _add(avg_vel, other.get("velocity", (0.0, 0.0)))
                avg_pos = _add(avg_pos, opos)
            alignment = _sub(_scale(avg_vel, 1.0 / len(neighbors)), vel)
            cohesion = _normalize(_sub(_scale(avg_pos, 1.0 / len(neighbors)), pos))

        # Leadership: follow the agent with the most leadership/priority.
        candidates = [agent] + list(neighbors)
        leader = max(
            candidates,
            key=lambda a: (a.get("leadership", 0.0), a.get("priority", 0.0)),
        )
        leadership_force = (0.0, 0.0)
        if leader is not agent:
            leadership_force = _scale(
                _normalize(leader.get("velocity", (0.0, 0.0))), leader_weight)

        # Goal direction: the crowd shares a common destination.
        goal_force = (0.0, 0.0)
        if goal is not None:
            goal_force = _scale(_normalize(_sub(goal, pos)), goal_weight)

        force = _add(
            _add(_add(_scale(separation, 1.5), _scale(alignment, 1.0)),
                 _scale(cohesion, 0.5)),
            _add(leadership_force, goal_force),
        )
        return {
            "force": force,
            "separation": separation,
            "alignment": alignment,
            "cohesion": cohesion,
            "leadership": leadership_force,
            "leader_id": _agent_id(leader),
            "is_leader": leader is agent,
            "goal_force": goal_force,
        }

    # ------------------------------------------------------------------
    # 5. Rendering: instancing + animation reuse + LOD
    # ------------------------------------------------------------------
    def render_crowd(self, agents=None, camera=(0.0, 0.0),
                     lod_distances=None):
        agents = agents if agents is not None else []
        lod_distances = lod_distances or self.DEFAULT_LOD_DISTANCES
        near, far = lod_distances

        batches = {}
        lod_assignment = {}
        for agent in agents:
            pos = agent.get("position", (0.0, 0.0))
            dist = _length(_sub(pos, camera))
            lod = "high" if dist <= near else ("medium" if dist <= far else "low")
            lod_assignment[_agent_id(agent)] = lod

            # Animation reuse: one shared clip per (state, lod); distant
            # agents fall back to a cheap static pose instead of full anim.
            anim_state = agent.get("anim_state", "walk") if lod != "low" else "static"
            clip_key = (anim_state, lod)
            if clip_key not in self._animation_cache:
                self._animation_cache[clip_key] = {
                    "state": anim_state,
                    "lod": lod,
                    "frames": 1 if lod == "low" else (8 if lod == "medium" else 24),
                }
            clip = self._animation_cache[clip_key]

            # Instancing: one draw call per (mesh, clip) batch.
            mesh = agent.get("mesh", "default")
            batch_key = (mesh, clip_key)
            batch = batches.setdefault(batch_key, {
                "mesh": mesh,
                "clip": clip,
                "lod": lod,
                "instances": [],
            })
            batch["instances"].append(_agent_id(agent))

        return {
            "instanced": True,
            "draw_calls": len(batches),
            "batches": list(batches.values()),
            "lod_assignment": lod_assignment,
            "animation_clips": len(self._animation_cache),
            "agent_count": len(agents),
        }
