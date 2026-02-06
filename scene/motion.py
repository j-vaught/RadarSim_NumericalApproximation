"""Motion / Path Engine — generates per-frame positions for all path types."""

import math
import warnings
from typing import Any, List, Optional, Tuple

import numpy as np

from .config import PathConfig
from ..core import NUM_RANGE_BINS

# Safe math namespace for eval expressions
_SAFE_MATH = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "sqrt": math.sqrt, "abs": abs, "pi": math.pi,
    "min": min, "max": max, "log": math.log, "exp": math.exp,
}


def safe_eval(expression: str, variables: dict) -> float:
    """Evaluate a math expression with whitelisted functions and variables."""
    ns = dict(_SAFE_MATH)
    ns.update(variables)
    try:
        return float(eval(expression, {"__builtins__": {}}, ns))
    except Exception as e:
        raise ValueError(f"Failed to evaluate expression '{expression}': {e}")


def eval_speed(speed_spec: Any, t: int) -> float:
    """Evaluate speed — either a constant number or an expression string."""
    if isinstance(speed_spec, (int, float)):
        return float(speed_spec)
    return safe_eval(str(speed_spec), {"t": t})


def resolve_position(spec, valid_positions, edge_positions, rng) -> Tuple[int, int]:
    """Resolve a position specification to (pulse, bin).

    spec: "random", "edge", or [pulse, bin]
    """
    if isinstance(spec, str):
        if spec == "random":
            return tuple(rng.choice(valid_positions))
        elif spec == "edge":
            if edge_positions:
                return tuple(rng.choice(edge_positions))
            return tuple(rng.choice(valid_positions))
        else:
            raise ValueError(f"Unknown position spec: {spec}")
    elif isinstance(spec, (list, tuple)) and len(spec) == 2:
        return (int(spec[0]), int(spec[1]))
    raise ValueError(f"Invalid position spec: {spec}")


def validate_edge_start(position: Tuple[int, int], edge_positions: List[Tuple[int, int]],
                        water_mask, num_pulses: int, tolerance: int = 10) -> bool:
    """Check if position is at an edge (water/land boundary or radar coverage boundary)."""
    p, b = position
    # Coverage boundary: near min/max range bins
    if b < 60 or b > NUM_RANGE_BINS - 60:
        return True
    # Check proximity to any edge position
    for ep, eb in edge_positions:
        dp = min(abs(p - ep), num_pulses - abs(p - ep))
        db = abs(b - eb)
        if dp <= tolerance and db <= tolerance:
            return True
    return False


def snap_to_water(pulse: int, bin_idx: int, water_mask, num_pulses: int,
                  max_radius: int = 20) -> Optional[Tuple[int, int]]:
    """Find nearest valid water pixel via expanding search."""
    for r in range(0, max_radius + 1):
        for dp in range(-r, r + 1):
            for db in range(-r, r + 1):
                if abs(dp) != r and abs(db) != r:
                    continue
                p = (pulse + dp) % num_pulses
                b = bin_idx + db
                if 0 <= b < NUM_RANGE_BINS and water_mask.get(p, b):
                    return (p, b)
    return None


def _validate_point(pulse: int, bin_idx: int, water_mask, num_pulses: int,
                    allow_land: bool = False) -> Optional[Tuple[int, int]]:
    """Validate a point against water mask.

    Args:
        pulse: Pulse index
        bin_idx: Range bin index
        water_mask: PolarMask for water regions
        num_pulses: Total number of pulses
        allow_land: If True, snap to water when on land. If False, return None.

    Returns:
        (pulse, bin_idx) tuple if valid, or None if on land and allow_land=False.
    """
    pulse = pulse % num_pulses
    bin_idx = max(0, min(bin_idx, NUM_RANGE_BINS - 1))
    if water_mask.get(pulse, bin_idx):
        return (pulse, bin_idx)
    if allow_land:
        result = snap_to_water(pulse, bin_idx, water_mask, num_pulses)
        if result:
            warnings.warn(f"Position ({pulse}, {bin_idx}) is on land, snapped to {result}")
            return result
        warnings.warn(f"Position ({pulse}, {bin_idx}) is on land, no nearby water found")
        return (pulse, bin_idx)
    return None


def generate_path(path_config: PathConfig, total_frames: int,
                  valid_positions: List[Tuple[int, int]],
                  edge_positions: List[Tuple[int, int]],
                  water_mask, num_pulses: int, rng,
                  allow_land: bool = False) -> Tuple[List[Tuple[int, int]], int, int]:
    """Generate a path according to config.

    Args:
        path_config: PathConfig from YAML
        total_frames: Total frames in sequence
        valid_positions: Safe positions within water
        edge_positions: Positions at water/land boundary
        water_mask: PolarMask for water regions
        num_pulses: Total number of pulses
        rng: Random number generator
        allow_land: If True, snap to water when on land. If False, truncate path.

    Returns: (positions, start_frame, end_frame)
    positions has one entry per active frame.
    """
    path_type = path_config.type
    # Use per-object allow_land from config, or CLI override
    effective_allow_land = path_config.allow_land or allow_land

    if path_type == "fixed":
        return _path_fixed(path_config, total_frames, valid_positions, edge_positions,
                           water_mask, num_pulses, rng, effective_allow_land)
    elif path_type == "linear":
        return _path_linear(path_config, total_frames, valid_positions, edge_positions,
                            water_mask, num_pulses, rng, effective_allow_land)
    elif path_type == "bezier":
        return _path_bezier(path_config, total_frames, valid_positions, edge_positions,
                            water_mask, num_pulses, rng, effective_allow_land)
    elif path_type == "waypoints":
        return _path_waypoints(path_config, total_frames, valid_positions, edge_positions,
                               water_mask, num_pulses, rng, effective_allow_land)
    elif path_type == "equation":
        return _path_equation(path_config, total_frames, valid_positions, edge_positions,
                              water_mask, num_pulses, rng, effective_allow_land)
    else:
        raise ValueError(f"Unknown path type: {path_type}")


def _compute_duration_frames(config: PathConfig, total_frames: int, rng) -> Tuple[int, int]:
    """Compute start_frame, end_frame from PathConfig.

    Priority:
    1. Both start_frame + end_frame set → use directly (clamped to total_frames)
    2. Only start_frame set → use duration for length, starting at start_frame
    3. Only end_frame set → use duration for length, ending at end_frame
    4. Neither set → original duration-based random logic
    """
    sf = config.start_frame
    ef = config.end_frame
    duration = config.duration

    if sf is not None and ef is not None:
        return max(0, sf), min(ef, total_frames)

    # Compute duration-based active length
    frac = rng.uniform(duration[0], duration[1]) if len(duration) == 2 else duration[0]
    active_frames = max(1, int(total_frames * frac))

    if sf is not None:
        start = max(0, sf)
        end = min(start + active_frames, total_frames)
        return start, end

    if ef is not None:
        end = min(ef, total_frames)
        start = max(0, end - active_frames)
        return start, end

    # Neither set — original random logic
    max_start = total_frames - active_frames
    start_frame = rng.randint(0, max(0, max_start))
    end_frame = min(start_frame + active_frames, total_frames)
    return start_frame, end_frame


def _path_fixed(config, total_frames, valid_positions, edge_positions,
                water_mask, num_pulses, rng, allow_land=False):
    pos = resolve_position(config.position, valid_positions, edge_positions, rng)
    validated = _validate_point(pos[0], pos[1], water_mask, num_pulses, allow_land=True)
    # Respect start_frame/end_frame if set, otherwise present for ALL frames
    if config.start_frame is not None or config.end_frame is not None:
        sf = max(0, config.start_frame or 0)
        ef = min(config.end_frame or total_frames, total_frames)
        active = ef - sf
        positions = [validated] * active
        return positions, sf, ef
    positions = [validated] * total_frames
    return positions, 0, total_frames


def _path_linear(config, total_frames, valid_positions, edge_positions,
                 water_mask, num_pulses, rng, allow_land=False):
    start = resolve_position(config.start, valid_positions, edge_positions, rng)

    # Validate edge start for moving objects
    if not validate_edge_start(start, edge_positions, water_mask, num_pulses):
        raise ValueError(
            f"Moving object start position {start} is not at an edge. "
            "Moving objects must start from an edge (water/land boundary or radar coverage boundary)."
        )

    start_frame, end_frame = _compute_duration_frames(config, total_frames, rng)
    active_frames = end_frame - start_frame

    heading_rad = math.radians(config.heading)
    # In polar space: pulse direction ~ angular, bin direction ~ radial
    # heading 0=N means decreasing bin (toward center), 90=E means increasing pulse
    dp_per_unit = math.sin(heading_rad)
    db_per_unit = -math.cos(heading_rad)

    positions = []
    cumulative_p = float(start[0])
    cumulative_b = float(start[1])

    for t in range(active_frames):
        spd = eval_speed(config.speed, t)
        if t > 0:
            cumulative_p += dp_per_unit * spd
            cumulative_b += db_per_unit * spd

        pulse = int(cumulative_p) % num_pulses
        bin_idx = max(0, min(int(cumulative_b), NUM_RANGE_BINS - 1))
        validated = _validate_point(pulse, bin_idx, water_mask, num_pulses, allow_land)
        if validated is None:
            break
        positions.append(validated)

    actual_end = start_frame + len(positions)
    return positions, start_frame, actual_end


def _path_bezier(config, total_frames, valid_positions, edge_positions,
                 water_mask, num_pulses, rng, allow_land=False):
    start = resolve_position(config.start, valid_positions, edge_positions, rng)
    end = resolve_position(config.end, valid_positions, edge_positions, rng)

    if not validate_edge_start(start, edge_positions, water_mask, num_pulses):
        raise ValueError(f"Bezier start {start} is not at an edge.")

    start_frame, end_frame = _compute_duration_frames(config, total_frames, rng)
    active_frames = end_frame - start_frame

    # Control point: perpendicular offset
    mid_p = (start[0] + end[0]) / 2.0
    mid_b = (start[1] + end[1]) / 2.0
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    length = max(math.sqrt(dx * dx + dy * dy), 1.0)
    perp_x = -dy / length
    perp_y = dx / length
    curve_sign = 1.0 if rng.random() < 0.5 else -1.0
    offset = config.curvature * length * 0.5 * curve_sign
    ctrl_p = mid_p + perp_x * offset
    ctrl_b = mid_b + perp_y * offset

    # Pre-compute arc length for speed-based parameterization
    num_samples = max(active_frames * 4, 200)
    raw_points = []
    for i in range(num_samples + 1):
        u = i / num_samples
        u1 = 1.0 - u
        p = u1 * u1 * start[0] + 2.0 * u1 * u * ctrl_p + u * u * end[0]
        b = u1 * u1 * start[1] + 2.0 * u1 * u * ctrl_b + u * u * end[1]
        raw_points.append((p, b))

    arc_lengths = [0.0]
    for i in range(1, len(raw_points)):
        dp = raw_points[i][0] - raw_points[i - 1][0]
        db = raw_points[i][1] - raw_points[i - 1][1]
        arc_lengths.append(arc_lengths[-1] + math.sqrt(dp * dp + db * db))
    total_arc = arc_lengths[-1]

    # Sample by speed
    positions = []
    dist = 0.0
    for t in range(active_frames):
        spd = eval_speed(config.speed, t)
        if t > 0:
            dist += spd
        if dist > total_arc:
            dist = total_arc

        # Find u for this distance
        idx = 0
        for j in range(1, len(arc_lengths)):
            if arc_lengths[j] >= dist:
                idx = j - 1
                break
        else:
            idx = len(arc_lengths) - 2

        seg_len = arc_lengths[idx + 1] - arc_lengths[idx]
        frac = (dist - arc_lengths[idx]) / seg_len if seg_len > 0 else 0.0
        p = raw_points[idx][0] + frac * (raw_points[idx + 1][0] - raw_points[idx][0])
        b = raw_points[idx][1] + frac * (raw_points[idx + 1][1] - raw_points[idx][1])

        pulse = int(p) % num_pulses
        bin_idx = max(0, min(int(b), NUM_RANGE_BINS - 1))
        validated = _validate_point(pulse, bin_idx, water_mask, num_pulses, allow_land)
        if validated is None:
            break
        positions.append(validated)

    actual_end = start_frame + len(positions)
    return positions, start_frame, actual_end


def _path_waypoints(config, total_frames, valid_positions, edge_positions,
                    water_mask, num_pulses, rng, allow_land=False):
    points = config.points
    if len(points) < 2:
        raise ValueError("Waypoints path requires at least 2 points")

    # Validate all waypoints are in water
    for pt in points:
        p, b = int(pt[0]), int(pt[1])
        if not water_mask.get(p % num_pulses, max(0, min(b, NUM_RANGE_BINS - 1))):
            raise ValueError(f"Waypoint ({p}, {b}) is on land. Fix the YAML.")

    first = (int(points[0][0]), int(points[0][1]))
    if not validate_edge_start(first, edge_positions, water_mask, num_pulses):
        raise ValueError(f"Waypoints start {first} is not at an edge.")

    start_frame, end_frame = _compute_duration_frames(config, total_frames, rng)
    active_frames = end_frame - start_frame

    # Build spline
    pts_p = [float(pt[0]) for pt in points]
    pts_b = [float(pt[1]) for pt in points]

    if config.loop:
        pts_p.append(pts_p[0])
        pts_b.append(pts_b[0])

    n = len(pts_p)

    if config.smoothing == "cubic" and n >= 4:
        raw_points = _cubic_spline_sample(pts_p, pts_b, max(active_frames * 4, 200))
    elif config.smoothing == "linear" or n < 4:
        raw_points = _linear_interp_sample(pts_p, pts_b, max(active_frames * 4, 200))
    else:
        raw_points = _linear_interp_sample(pts_p, pts_b, max(active_frames * 4, 200))

    # Arc-length parameterization
    arc_lengths = [0.0]
    for i in range(1, len(raw_points)):
        dp = raw_points[i][0] - raw_points[i - 1][0]
        db = raw_points[i][1] - raw_points[i - 1][1]
        arc_lengths.append(arc_lengths[-1] + math.sqrt(dp * dp + db * db))
    total_arc = arc_lengths[-1]

    positions = []
    dist = 0.0
    for t in range(active_frames):
        spd = eval_speed(config.speed, t)
        if t > 0:
            dist += spd
        if config.loop and dist > total_arc:
            dist = dist % total_arc if total_arc > 0 else 0.0
        elif dist > total_arc:
            dist = total_arc

        idx = 0
        for j in range(1, len(arc_lengths)):
            if arc_lengths[j] >= dist:
                idx = j - 1
                break
        else:
            idx = len(arc_lengths) - 2

        seg_len = arc_lengths[idx + 1] - arc_lengths[idx]
        frac = (dist - arc_lengths[idx]) / seg_len if seg_len > 0 else 0.0
        p = raw_points[idx][0] + frac * (raw_points[idx + 1][0] - raw_points[idx][0])
        b = raw_points[idx][1] + frac * (raw_points[idx + 1][1] - raw_points[idx][1])

        pulse = int(p) % num_pulses
        bin_idx = max(0, min(int(b), NUM_RANGE_BINS - 1))
        validated = _validate_point(pulse, bin_idx, water_mask, num_pulses, allow_land)
        if validated is None:
            break
        positions.append(validated)

    actual_end = start_frame + len(positions)
    return positions, start_frame, actual_end


def _linear_interp_sample(pts_p, pts_b, num_samples):
    """Linearly interpolate between waypoints."""
    n = len(pts_p)
    result = []
    for i in range(num_samples + 1):
        t = i / num_samples * (n - 1)
        idx = min(int(t), n - 2)
        frac = t - idx
        p = pts_p[idx] + frac * (pts_p[idx + 1] - pts_p[idx])
        b = pts_b[idx] + frac * (pts_b[idx + 1] - pts_b[idx])
        result.append((p, b))
    return result


def _cubic_spline_sample(pts_p, pts_b, num_samples):
    """Cubic spline interpolation through waypoints using numpy."""
    n = len(pts_p)
    # Natural cubic spline for each coordinate
    result = []
    ts = list(range(n))
    t_interp = [i / num_samples * (n - 1) for i in range(num_samples + 1)]

    p_interp = _natural_cubic_spline(ts, pts_p, t_interp)
    b_interp = _natural_cubic_spline(ts, pts_b, t_interp)

    for p, b in zip(p_interp, b_interp):
        result.append((p, b))
    return result


def _natural_cubic_spline(xs, ys, xs_interp):
    """Simple natural cubic spline implementation."""
    n = len(xs)
    if n < 2:
        return [ys[0]] * len(xs_interp)
    if n == 2:
        return [ys[0] + (ys[1] - ys[0]) * (x - xs[0]) / (xs[1] - xs[0]) for x in xs_interp]
    if n == 3:
        # Quadratic fallback
        return _linear_interp_vals(xs, ys, xs_interp)

    h = [xs[i + 1] - xs[i] for i in range(n - 1)]
    alpha = [0.0] * n
    for i in range(1, n - 1):
        alpha[i] = (3.0 / h[i] * (ys[i + 1] - ys[i]) - 3.0 / h[i - 1] * (ys[i] - ys[i - 1]))

    l = [1.0] * n
    mu = [0.0] * n
    z = [0.0] * n
    for i in range(1, n - 1):
        l[i] = 2.0 * (xs[i + 1] - xs[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    c = [0.0] * n
    b_coef = [0.0] * n
    d = [0.0] * n
    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b_coef[j] = (ys[j + 1] - ys[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0
        d[j] = (c[j + 1] - c[j]) / (3.0 * h[j])

    result = []
    for x in xs_interp:
        idx = min(max(0, int(x)), n - 2)
        for k in range(n - 2, -1, -1):
            if xs[k] <= x:
                idx = k
                break
        dx = x - xs[idx]
        val = ys[idx] + b_coef[idx] * dx + c[idx] * dx * dx + d[idx] * dx * dx * dx
        result.append(val)
    return result


def _linear_interp_vals(xs, ys, xs_interp):
    n = len(xs)
    result = []
    for x in xs_interp:
        idx = min(max(0, int(x)), n - 2)
        for k in range(n - 2, -1, -1):
            if xs[k] <= x:
                idx = k
                break
        frac = (x - xs[idx]) / (xs[idx + 1] - xs[idx]) if xs[idx + 1] != xs[idx] else 0.0
        result.append(ys[idx] + frac * (ys[idx + 1] - ys[idx]))
    return result


def _path_equation(config, total_frames, valid_positions, edge_positions,
                   water_mask, num_pulses, rng, allow_land=False):
    if not config.pulse_expr or not config.bin_expr:
        raise ValueError("Equation path requires 'pulse' and 'bin' expressions")

    start_frame, end_frame = _compute_duration_frames(config, total_frames, rng)
    active_frames = end_frame - start_frame

    center_pulse = num_pulses // 2
    center_bin = NUM_RANGE_BINS // 2
    variables_base = {
        "center_pulse": center_pulse,
        "center_bin": center_bin,
        "num_pulses": num_pulses,
        "num_bins": NUM_RANGE_BINS,
    }

    positions = []
    for t in range(active_frames):
        variables = dict(variables_base)
        variables["t"] = t
        p = safe_eval(config.pulse_expr, variables)
        b = safe_eval(config.bin_expr, variables)
        pulse = int(p) % num_pulses
        bin_idx = max(0, min(int(b), NUM_RANGE_BINS - 1))
        validated = _validate_point(pulse, bin_idx, water_mask, num_pulses, allow_land)
        if validated is None:
            break
        positions.append(validated)

    actual_end = start_frame + len(positions)
    return positions, start_frame, actual_end
