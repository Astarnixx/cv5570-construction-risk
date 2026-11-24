"""risk_pose.py - shared pose logic (single source of truth).
Written to Drive by the pose notebook, imported by the engines notebook.

What it does
  * run_pose            full-frame pose (all people, 17 COCO joints each)
  * merge_detector_persons   one worker list: pose people + detector 'Person' boxes the pose model missed
  * posture             torso lean / balance / 'down' state from the joints
  * analyse_persons     per-worker ELEVATION above the ground: every upright full-body worker is a 1.7 m ruler that gives
                        the camera height above the plane THEY stand on; the biggest consistent group is the ground,
                        anyone whose camera height is smaller is standing higher up.
Conventions: pixels are full-image pixels; camera frame is x right, y down, z forward; GeoCalib's saved 'g' is the UP vector.
"""
import math
import numpy as np

COCO = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow',
        'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee',
        'left_ankle', 'right_ankle']
L_SH, R_SH, L_HIP, R_HIP, L_ANK, R_ANK = 5, 6, 11, 12, 15, 16
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (0, 1), (0, 2), (1, 3), (2, 4)]

CFG = dict(
    POSE_MODEL='yolov8m-pose.pt', POSE_IMGSZ=1280, POSE_CONF=0.25,
    KP_CONF=0.3,            # a joint counts as visible above this (posture)
    ANKLE_CONF=0.5,         # both ankles must exceed this for a worker to be used as a height ruler / to place the feet
    MATCH_IOU=0.4,          # pose person and detector person are the same worker above this IoU
    # --- camera / ruler ---
    WORKER_H=1.7, MIN_ASPECT=1.3, EDGE_MARGIN=0.01, TOPDOWN_G=0.85, HORIZON_DEN=-0.01, MIN_DEN=0.01,
    MIN_BOX_FRAC=0.04,      # a ruler worker must be at least this tall (fraction of image height)
    MAX_RANGE_M=20.0,       # and not further than this (a camera-tilt error of 1 deg moves the estimate ~0.017 m per metre of range)
    # --- ground reference and elevation ---
    CLUSTER_ABS=0.25, CLUSTER_REL=0.15,     # camera heights within max(0.25 m, 15%) count as the same plane
    MIN_REF=2,                              # ground workers needed before any elevation verdict (unless camera height is known)
    ELEV_MIN_M=0.5,                         # at least this far above the ground = 'elevated'
    ELEV_FULL_M=1.8,                        # severity 1.0 from here (fall protection is required from 6 ft = 1.8 m)
    # --- posture ---
    LEAN_START=20, LEAN_RANGE=50, COM_RANGE=0.15, DOWN_LEAN_MIN=55, DOWN_ASPECT_NO_KP=1.3,
)

# ------------------------------------------------------------------ pose + merging
def load_pose_model(name=None):
    from ultralytics import YOLO
    return YOLO(name or CFG['POSE_MODEL'])

def _cx(p):
    return (p['box'][0] + p['box'][2]) / 2

def run_pose(model, img):
    """full-frame pose -> list of dict(box, conf, kp (17x3), src='pose'), sorted left to right"""
    r = model.predict(img, imgsz=CFG['POSE_IMGSZ'], conf=CFG['POSE_CONF'], verbose=False)[0]
    out = []
    if r.boxes is not None and len(r.boxes):
        kps = r.keypoints.data.cpu().numpy()
        for i, b in enumerate(r.boxes):
            out.append(dict(box=[float(v) for v in b.xyxy[0]], conf=float(b.conf), kp=np.asarray(kps[i], float), src='pose'))
    return sorted(out, key=_cx)

def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0

def merge_detector_persons(persons, det_boxes, det_confs=None):
    """One worker list: pose people (with joints) + detector people the pose model missed (no joints).
    One-to-one matching, highest IoU first. src is 'pose', 'both' or 'det'."""
    persons = [dict(p) for p in persons]
    pairs = sorted(((iou(p['box'], b), i, j) for i, p in enumerate(persons) for j, b in enumerate(det_boxes)), reverse=True)
    used_p, used_d = set(), set()
    for v, i, j in pairs:
        if v < CFG['MATCH_IOU']: break
        if i in used_p or j in used_d: continue
        used_p.add(i); used_d.add(j)
        persons[i]['src'] = 'both'; persons[i]['det_box'] = [float(x) for x in det_boxes[j]]
    for j, b in enumerate(det_boxes):
        if j not in used_d:
            persons.append(dict(box=[float(x) for x in b], conf=None if det_confs is None else float(det_confs[j]), kp=None, src='det'))
    return sorted(persons, key=_cx)

def feet_px(p):
    """middle of the two ankles if both are visible, else bottom-centre of the box"""
    kp = p.get('kp')
    if kp is not None:
        a = [kp[i][:2] for i in (L_ANK, R_ANK) if kp[i][2] > CFG['ANKLE_CONF']]
        if a: return tuple(float(v) for v in np.mean(a, axis=0))
    return ((p['box'][0] + p['box'][2]) / 2, p['box'][3])

# ------------------------------------------------------------------ posture
def posture(p):
    """torso lean (0 upright, 90 horizontal, >90 inverted), balance (hip centre outside the ankles, as a fraction of body
    height) and a 'down' state (box wider than tall) -> value 0..1 and a label"""
    res = dict(label='unknown', value=None, lean_deg=None, balance=None, down=False)
    kp = p.get('kp')
    if kp is None: return res
    x1, y1, x2, y2 = p['box']; bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    pt = lambda i: kp[i][:2] if kp[i][2] > CFG['KP_CONF'] else None
    sh = [q for q in (pt(L_SH), pt(R_SH)) if q is not None]
    hp = [q for q in (pt(L_HIP), pt(R_HIP)) if q is not None]
    an = [q for q in (pt(L_ANK), pt(R_ANK)) if q is not None]
    lean_t = com_t = None
    if sh and hp:
        v = np.mean(sh, axis=0) - np.mean(hp, axis=0)
        lean = math.degrees(math.atan2(abs(v[0]), -v[1]))
        res['lean_deg'] = float(lean)
        lean_t = float(np.clip((lean - CFG['LEAN_START']) / CFG['LEAN_RANGE'], 0, 1))
    if hp and an:
        hx = np.mean(hp, axis=0)[0]; xs = [q[0] for q in an]
        off = max(0.0, min(xs) - hx, hx - max(xs)) / bh
        res['balance'] = float(off)
        com_t = float(np.clip(off / CFG['COM_RANGE'], 0, 1))
    parts = [t for t in (lean_t, None if com_t is None else 0.7 * com_t) if t is not None]
    # 'down' = lying: box wider than tall AND torso near horizontal. A crouching / seated person also has a wide box,
    # but an upright-ish torso, so they are scored normally (lean / balance) and not as 'down'.
    lying = bw / bh > 1.0 and (res['lean_deg'] is None and bw / bh > CFG['DOWN_ASPECT_NO_KP'] or
                               res['lean_deg'] is not None and res['lean_deg'] >= CFG['DOWN_LEAN_MIN'])
    if lying:
        res.update(value=1.0, label='down', down=True)
    elif parts:
        v_ = max(parts); res.update(value=float(v_), label='leaning' if v_ > 0.4 else 'upright')
    return res

def head_px(p):
    """approximate head height in pixels, used to decide whether a helmet could be seen at all.
    From joints (nose to the shoulder line x 1.35) when available, otherwise 13% of the person-box height."""
    kp = p.get('kp'); bh = max(1.0, p['box'][3] - p['box'][1])
    if kp is not None and kp[0][2] > CFG['KP_CONF'] and kp[5][2] > CFG['KP_CONF'] and kp[6][2] > CFG['KP_CONF']:
        sh = (np.asarray(kp[5][:2]) + np.asarray(kp[6][:2])) / 2
        d = float(np.hypot(*(np.asarray(kp[0][:2]) - sh)))
        if d > 2: return 1.35 * d
    return 0.13 * bh

# ------------------------------------------------------------------ camera helpers (GeoCalib json)
def g_down(cam):
    g = -np.array(cam['g'], float)             # GeoCalib's 'g' is the UP vector; the floor geometry needs DOWN
    return g / np.linalg.norm(g)

def is_topdown(cam):
    return g_down(cam)[2] > CFG['TOPDOWN_G']

def ray(cam, u, v):
    return np.array([(u - cam['cx']) / cam['fx'], (v - cam['cy']) / cam['fy'], 1.0])

def floor_point(u, v, cam, h):
    """3D point (m, camera frame) where the pixel's ray meets the plane that lies h below the camera; None at/above the horizon"""
    d = ray(cam, u, v); den = g_down(cam) @ d
    return None if den <= 0.02 else d * h / den

def horizon_line(cam, width):
    """two image points of the horizon (where the plane through the camera centre meets the image) or None"""
    g = g_down(cam)
    if abs(g[1]) < 1e-3: return None
    v = lambda u: cam['cy'] - cam['fy'] * (g[2] + g[0] * (u - cam['cx']) / cam['fx']) / g[1]
    return (0.0, float(v(0.0))), (float(width), float(v(float(width))))

# ------------------------------------------------------------------ elevation
def person_height(p, cam, shape):
    """Camera height above the plane THIS worker stands on, assuming WORKER_H tall.
    foot = h*a (on the plane), head = foot - WORKER_H*g_down = s*d_head  ->  solve for (s, h).  Returns (h or None, status)."""
    kp = p.get('kp')
    if kp is None: return None, 'no_keypoints'
    if not (kp[L_ANK][2] > CFG['ANKLE_CONF'] and kp[R_ANK][2] > CFG['ANKLE_CONF']): return None, 'ankles_not_visible'
    x1, y1, x2, y2 = p['box']; H = shape[0]
    if y1 < CFG['EDGE_MARGIN'] * H or y2 > (1 - CFG['EDGE_MARGIN']) * H: return None, 'clipped_by_image_edge'
    if (y2 - y1) < CFG['MIN_ASPECT'] * (x2 - x1): return None, 'not_upright'
    if (y2 - y1) < CFG['MIN_BOX_FRAC'] * H: return None, 'too_small'
    if cam['fy'] * CFG['WORKER_H'] / (y2 - y1) > CFG['MAX_RANGE_M']: return None, 'too_far'
    g = g_down(cam)
    d_f = ray(cam, *feet_px(p)); d_t = ray(cam, (x1 + x2) / 2, y1)
    den = g @ d_f
    if den < CFG['HORIZON_DEN']: return None, 'above_horizon'
    if den < CFG['MIN_DEN']: return None, 'near_horizon'
    a = d_f / den
    sol, *_ = np.linalg.lstsq(np.stack([d_t, -a], axis=1), -CFG['WORKER_H'] * g, rcond=None)
    s, h = float(sol[0]), float(sol[1])
    if s <= 0 or h <= 0.02: return None, 'bad_solution'       # h may be small: a platform just below the camera's eye level
    return h, 'ok'

def _ground_cluster(vals):
    """largest group of consistent camera heights = the ground plane (tie -> the larger height: the ground is the lowest plane)"""
    vals = np.asarray(vals, float); best = None
    for v in vals:
        m = np.abs(vals - v) <= max(CFG['CLUSTER_ABS'], CFG['CLUSTER_REL'] * v)
        key = (int(m.sum()), float(v))
        if best is None or key > best[0]: best = (key, m)
    return best[1]

def analyse_persons(persons, cam, shape, h_known=None):
    """Adds p['posture'] and p['elev'] to every person (in place) and returns a scene summary.
    elev kinds: ground | elevated | above_camera (feet at/above the camera's eye level: elevation ~ camera height or more, m = camera height)
                | below_ground | unknown.  Verdicts need >= MIN_REF ground workers, or a known camera height."""
    scene = dict(h_ref=None, spread=None, n_ref=0, ref_source=None, elev_ok=False, conf=None, status='', ground_idx=[])
    for p in persons:
        p['posture'] = posture(p)
        p['elev'] = dict(kind='unknown', m=None, sev=None, conf=None, h_i=None, status='')
    if cam is None:
        scene['status'] = 'no camera (GeoCalib) file'; return scene
    if is_topdown(cam):
        scene['status'] = 'camera looks almost straight down'; return scene

    hs = []
    for i, p in enumerate(persons):
        h, st = person_height(p, cam, shape)
        p['elev']['h_i'], p['elev']['status'] = h, st
        if h is not None: hs.append((i, h))

    if h_known:
        scene.update(h_ref=float(h_known), ref_source='known camera height', elev_ok=True, conf='high')
        ground = [i for i, h in hs if abs(h - h_known) <= max(CFG['CLUSTER_ABS'], CFG['CLUSTER_REL'] * h_known)]
        scene.update(n_ref=len(ground), ground_idx=ground)
    elif hs:
        m = _ground_cluster([h for _, h in hs])
        ground = [hs[k][0] for k in range(len(hs)) if m[k]]; gh = np.array([hs[k][1] for k in range(len(hs)) if m[k]])
        med = float(np.median(gh))
        spread = float(np.median(np.abs(gh - med)) / med) if len(gh) > 1 else None
        scene.update(h_ref=med, spread=spread, n_ref=len(gh), ref_source='ground workers as 1.7 m rulers', ground_idx=ground)
        if len(gh) >= CFG['MIN_REF']:
            scene['elev_ok'] = True
            scene['conf'] = 'high' if (len(gh) >= 3 and spread is not None and spread <= 0.15) else 'medium'
        else:
            scene['status'] = 'only one full-body worker for the ground reference -> elevation not judged'
    else:
        scene['status'] = 'no upright full-body worker with both ankles visible'
        # a worker above the camera's eye level is still visible as 'above_horizon', but its size is unknown without a ruler
        return scene

    if not scene['elev_ok']: return scene
    for p in persons:
        e = p['elev']; e['conf'] = scene['conf']
        if e['h_i'] is not None:
            E = scene['h_ref'] - e['h_i']; e['m'] = float(E)
            if E >= CFG['ELEV_MIN_M']:
                e['kind'], e['sev'] = 'elevated', float(min(1.0, E / CFG['ELEV_FULL_M']))
            elif E <= -CFG['ELEV_MIN_M']:
                e['kind'], e['sev'] = 'below_ground', 0.0
            else:
                e['kind'], e['sev'] = 'ground', 0.0
        elif e['status'] in ('above_horizon', 'near_horizon'):      # feet at or above the camera's eye level
            e['kind'], e['m'] = 'above_camera', float(scene['h_ref'])
            e['sev'] = float(min(1.0, scene['h_ref'] / CFG['ELEV_FULL_M']))
    scene['status'] = 'ok'
    return scene

def plane_height(p, scene):
    """camera height above the plane the worker's feet are on (their own plane if elevated, else the ground)"""
    e = p.get('elev') or {}
    if e.get('kind') == 'elevated' and e.get('h_i') is not None: return e['h_i']
    return scene.get('h_ref')

# ------------------------------------------------------------------ json (Drive) helpers
def _num(x):
    return None if x is None else float(x)

def persons_to_json(persons):
    out = []
    for i, p in enumerate(persons):
        out.append(dict(idx=i, src=p.get('src'), conf=_num(p.get('conf')), box=[float(v) for v in p['box']],
                        kp=None if p.get('kp') is None else np.asarray(p['kp'], float).round(2).tolist(),
                        posture=p.get('posture'), elevation=p.get('elev')))
    return out

def raw_persons_from_json(d):
    """the expensive part only (box, conf, joints) of pose people; posture / elevation are always recomputed"""
    out = []
    for q in d['persons']:
        if q.get('kp') is None or q.get('src') == 'det': continue
        out.append(dict(box=q['box'], conf=q.get('conf'), kp=np.array(q['kp'], float), src='pose'))
    return sorted(out, key=_cx)
