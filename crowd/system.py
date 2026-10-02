"""pairwise-crowd-system - CrowdSystem

Crowd simulation with all-direction predictive avoidance, density control,
individual variation, smooth path following, full flocking (separation /
alignment / cohesion + leadership + goal seeking) and instanced rendering
with animation reuse and LOD.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

Vector = tuple[float, float]


# --- 2D vector helpers -----------------------------------------------------

def _add(a: Vector, b: Vector) -> Vector:
    return (a[0] + b[0], a[1] + b[1])


def _sub(a: Vector, b: Vector) -> Vector:
    return (a[0] - b[0], a[1] - b[1])


def _scale(a: Vector, s: float) -> Vector:
    return (a[0] * s, a[1] * s)


def _dot(a: Vector, b: Vector) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _length(a: Vector) -> float:
    return math.hypot(a[0], a[1])


def _normalize(a: Vector) -> Vector:
    size = _length(a)
    if size < 1e-9:
        return (0.0, 0.0)
    return (a[0] / size, a[1] / size)


def _clamp_length(a: Vector, max_len: float) -> Vector:
    size = _length(a)
    if size <= max_len or size < 1e-9:
        return a
    return _scale(a, max_len / size)


@dataclass
class Agent:
    """A single pedestrian with individual characteristics."""

    id: int
    position: Vector
    velocity: Vector = (0.0, 0.0)
    radius: float = 0.3
    base_speed: float = 1.4
    priority: int = 1  # higher priority wins right-of-way (e.g. elderly = 2)
    is_leader: bool = False
    path: list[Vector] = field(default_factory=list)
    speed_variation: float = 1.0  # individual speed difference
    animation_phase: float = 0.0  # desynchronised animation offset
    mesh_id: str = "citizen"


class CrowdSystem:
    PERCEPTION_RADIUS = 4.0      # avoidance scans ALL directions in this radius
    SEPARATION_RADIUS = 1.2      # personal space used by density control
    MAX_DENSITY = 3.0            # comfortable agents per m^2
    MAX_FORCE = 8.0
    PREDICTION_HORIZON = 2.0     # seconds of predictive avoidance
    LOD_DISTANCES = ((20.0, "high"), (50.0, "medium"), (float("inf"), "low"))

    def __init__(self, agents: list[Agent] | None = None, goal: Vector = (0.0, 0.0)):
        self.state: dict = {}
        self.goal = goal
        self._animation_cache: dict = {}
        self.agents = agents if agents is not None else self._spawn_demo_crowd()
        self.state["agents"] = self.agents

    # ------------------------------------------------------------------
    # crowd update: avoidance + density control + individual variation
    # ------------------------------------------------------------------
    def update_crowd(self, dt: float = 1.0 / 60.0,
                     agents: list[Agent] | None = None,
                     goal: Vector | None = None) -> list[Agent]:
        agents = agents if agents is not None else self.agents
        goal = self.goal if goal is None else goal
        for agent in agents:
            neighbors = self._neighbors_of(agent, agents)
            density = self._local_density(agent, neighbors)

            path = self.follow_path(agent, density=density)
            avoidance = self.calculate_avoidance(agent, neighbors)
            flock = self.calculate_flocking(agent, neighbors, goal=goal)["combined"]

            steering = _add(_add(path["steering"], avoidance), flock)
            steering = _clamp_length(steering, self.MAX_FORCE)

            # individual variation: every agent has its own top speed and
            # its own animation phase, so the crowd no longer moves in sync
            max_speed = agent.base_speed * agent.speed_variation * path["speed_factor"]
            velocity = _add(agent.velocity, _scale(steering, dt))
            agent.velocity = _clamp_length(velocity, max_speed)
            agent.position = _add(agent.position, _scale(agent.velocity, dt))
            agent.animation_phase = (agent.animation_phase
                                     + dt * _length(agent.velocity)) % 1.0

        self.state["agents"] = agents
        return agents

    # ------------------------------------------------------------------
    # avoidance: all directions + prediction + priority
    # ------------------------------------------------------------------
    def calculate_avoidance(self, agent: Agent | None = None,
                            neighbors: list[Agent] | None = None,
                            prediction_horizon: float | None = None) -> Vector:
        agent, neighbors = self._resolve(agent, neighbors)
        horizon = self.PREDICTION_HORIZON if prediction_horizon is None else prediction_horizon

        steering = (0.0, 0.0)
        for other in neighbors:
            ttc = self._time_to_collision(agent, other)
            distance = _length(_sub(other.position, agent.position))
            overlap = distance < agent.radius + other.radius
            if ttc is None and not overlap:
                continue
            if ttc is not None and ttc > horizon and not overlap:
                continue

            # predict where the other agent will be at collision time and
            # steer sideways away from it - works for front, side and rear
            t = ttc if ttc is not None else 0.0
            predicted = _add(other.position, _scale(other.velocity, t))
            away = _normalize(_sub(agent.position, predicted))
            if away == (0.0, 0.0):
                away = _normalize((-agent.velocity[1], agent.velocity[0]))

            urgency = 1.0 / max(t, 0.1)
            # priority: fully yield to higher-priority agents (elderly),
            # only partially to equals, barely to lower-priority ones
            if other.priority > agent.priority:
                weight = 1.0
            elif other.priority == agent.priority:
                weight = 0.6
            else:
                weight = 0.25
            steering = _add(steering, _scale(away, urgency * weight))

        return _clamp_length(steering, self.MAX_FORCE)

    # ------------------------------------------------------------------
    # path following: whole-path lookahead + smoothing + density aware
    # ------------------------------------------------------------------
    def follow_path(self, agent: Agent | None = None,
                    path: list[Vector] | None = None,
                    density: float = 0.0) -> dict:
        agent = agent if agent is not None else self.agents[0]
        path = path if path is not None else (agent.path or self._default_path())
        if len(path) < 2:
            return {"steering": (0.0, 0.0), "target": agent.position,
                    "speed_factor": 1.0}

        # project the position onto EVERY segment of the path instead of
        # only chasing the next waypoint
        closest_point, seg_index, seg_t = self._project_on_path(agent.position, path)
        seg_dir = _normalize(_sub(path[seg_index + 1], path[seg_index]))

        # lookahead along the path, proportional to current speed
        lookahead = max(agent.base_speed * agent.speed_variation * 1.5, 1.0)
        target = self._advance_along_path(path, seg_index, seg_t, lookahead)
        desired = _normalize(_sub(target, agent.position))

        # smoothing: slow down for sharp upcoming corners instead of
        # snapping direction at waypoints
        next_dir = seg_dir
        if seg_index + 2 < len(path):
            next_dir = _normalize(_sub(path[seg_index + 2], path[seg_index + 1]))
        turn_angle = math.acos(max(-1.0, min(1.0, _dot(seg_dir, next_dir))))
        corner_factor = max(0.3, 1.0 - turn_angle / math.pi)

        # density control: congested areas force everyone to slow down
        density_factor = 1.0 / (1.0 + max(0.0, density - self.MAX_DENSITY))
        speed_factor = corner_factor * density_factor

        target_speed = agent.base_speed * agent.speed_variation * speed_factor
        steering = _sub(_scale(desired, target_speed), agent.velocity)
        return {"steering": _clamp_length(steering, self.MAX_FORCE),
                "target": target, "speed_factor": speed_factor}

    # ------------------------------------------------------------------
    # flocking: separation + alignment + cohesion + leadership + goal
    # ------------------------------------------------------------------
    def calculate_flocking(self, agent: Agent | None = None,
                           neighbors: list[Agent] | None = None,
                           goal: Vector | None = None) -> dict:
        agent, neighbors = self._resolve(agent, neighbors)
        goal = self.goal if goal is None else goal

        separation = (0.0, 0.0)
        alignment = (0.0, 0.0)
        cohesion = (0.0, 0.0)
        leadership = (0.0, 0.0)

        if neighbors:
            centre = (0.0, 0.0)
            for other in neighbors:
                offset = _sub(agent.position, other.position)
                dist = max(_length(offset), 1e-6)
                # separation: push apart, stronger when closer
                separation = _add(separation, _scale(offset, 1.0 / (dist * dist)))
                alignment = _add(alignment, other.velocity)
                centre = _add(centre, other.position)
            count = float(len(neighbors))
            alignment = _sub(_scale(alignment, 1.0 / count), agent.velocity)
            cohesion = _sub(_scale(centre, 1.0 / count), agent.position)

            # leadership: follow the nearest leader's motion if one exists
            leaders = [n for n in neighbors if n.is_leader]
            if leaders:
                leader = min(leaders, key=lambda n: _length(
                    _sub(n.position, agent.position)))
                leadership = _add(
                    _scale(_normalize(_sub(leader.position, agent.position)), 0.5),
                    _scale(_normalize(leader.velocity), 0.5))

        # goal oriented: the whole crowd shares a common destination
        goal_force = _normalize(_sub(goal, agent.position))

        combined = _add(
            _add(_scale(separation, 1.5), _scale(_normalize(alignment), 1.0)),
            _add(_add(_scale(_normalize(cohesion), 0.8), leadership),
                 _scale(goal_force, 1.2)))
        return {
            "separation": separation,
            "alignment": alignment,
            "cohesion": cohesion,
            "leadership": leadership,
            "goal": goal_force,
            "combined": _clamp_length(combined, self.MAX_FORCE),
        }

    # ------------------------------------------------------------------
    # rendering: instancing + animation reuse + LOD
    # ------------------------------------------------------------------
    def render_crowd(self, agents: list[Agent] | None = None,
                     camera: Vector = (0.0, 0.0)) -> dict:
        agents = agents if agents is not None else self.agents

        batches: dict = {}
        for agent in agents:
            distance = _length(_sub(agent.position, camera))
            lod = self._lod_for_distance(distance)
            state = "walk" if _length(agent.velocity) > 0.1 else "idle"

            # animation reuse: one shared clip per (mesh, state, lod),
            # agents only carry a phase offset into the shared clip
            clip_key = (agent.mesh_id, state, lod)
            clip = self._animation_cache.setdefault(clip_key, {
                "mesh": agent.mesh_id, "state": state, "lod": lod,
                "frames": 1 if lod == "low" else (8 if lod == "medium" else 24),
            })

            # instancing: one draw call per (mesh, lod, clip) batch
            batch_key = (agent.mesh_id, lod, clip_key)
            batches.setdefault(batch_key, []).append({
                "agent_id": agent.id,
                "position": agent.position,
                "animation_phase": agent.animation_phase,
            })

        draw_calls = [{
            "mesh": key[0],
            "lod": key[1],
            "clip": self._animation_cache[key[2]],
            "instances": instances,
        } for key, instances in batches.items()]

        return {
            "draw_calls": draw_calls,
            "batch_count": len(draw_calls),
            "instance_count": len(agents),
            "animation_clips": list(self._animation_cache.values()),
        }

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _resolve(self, agent, neighbors):
        if agent is None:
            agent = self.agents[0]
        if neighbors is None:
            neighbors = self._neighbors_of(agent, self.agents)
        return agent, neighbors

    def _neighbors_of(self, agent: Agent, agents: list[Agent]) -> list[Agent]:
        # 360 degree scan: front, sides and rear all count
        return [other for other in agents
                if other is not agent
                and _length(_sub(other.position, agent.position))
                <= self.PERCEPTION_RADIUS]

    def _local_density(self, agent: Agent, neighbors: list[Agent]) -> float:
        close = [n for n in neighbors
                 if _length(_sub(n.position, agent.position))
                 <= self.SEPARATION_RADIUS]
        area = math.pi * self.SEPARATION_RADIUS ** 2
        return (len(close) + 1) / area

    @staticmethod
    def _time_to_collision(agent: Agent, other: Agent) -> float | None:
        rel_pos = _sub(other.position, agent.position)
        rel_vel = _sub(other.velocity, agent.velocity)
        radius = agent.radius + other.radius
        a = _dot(rel_vel, rel_vel)
        if a < 1e-9:
            return 0.0 if _length(rel_pos) <= radius else None
        b = 2.0 * _dot(rel_pos, rel_vel)
        c = _dot(rel_pos, rel_pos) - radius * radius
        if c <= 0.0:
            return 0.0
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return None
        t = (-b - math.sqrt(disc)) / (2.0 * a)
        return t if t >= 0.0 else None

    @staticmethod
    def _project_on_path(position: Vector, path: list[Vector]):
        best_point, best_index, best_t = path[0], 0, 0.0
        best_dist = float("inf")
        for i in range(len(path) - 1):
            start, end = path[i], path[i + 1]
            seg = _sub(end, start)
            seg_len_sq = max(_dot(seg, seg), 1e-9)
            t = max(0.0, min(1.0, _dot(_sub(position, start), seg) / seg_len_sq))
            point = _add(start, _scale(seg, t))
            dist = _length(_sub(position, point))
            if dist < best_dist:
                best_dist, best_point, best_index, best_t = dist, point, i, t
        return best_point, best_index, best_t

    @staticmethod
    def _advance_along_path(path: list[Vector], seg_index: int,
                            seg_t: float, distance: float) -> Vector:
        i, t, remaining = seg_index, seg_t, distance
        while i < len(path) - 1 and remaining > 0.0:
            start, end = path[i], path[i + 1]
            seg = _sub(end, start)
            seg_len = _length(seg)
            left = seg_len * (1.0 - t)
            if remaining <= left:
                return _add(start, _scale(seg, t + remaining / max(seg_len, 1e-9)))
            remaining -= left
            i, t = i + 1, 0.0
        return path[-1]

    def _lod_for_distance(self, distance: float) -> str:
        for max_distance, lod in self.LOD_DISTANCES:
            if distance <= max_distance:
                return lod
        return "low"

    @staticmethod
    def _default_path() -> list[Vector]:
        return [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]

    def _spawn_demo_crowd(self) -> list[Agent]:
        rng = random.Random(42)
        path = self._default_path()
        agents = []
        for i in range(12):
            agents.append(Agent(
                id=i,
                position=(rng.uniform(-4.0, 4.0), rng.uniform(-4.0, 4.0)),
                velocity=(rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5)),
                priority=2 if i % 5 == 0 else 1,  # some elderly pedestrians
                is_leader=(i == 0),
                path=list(path),
                speed_variation=rng.uniform(0.85, 1.15),
                animation_phase=rng.random(),
            ))
        return agents
