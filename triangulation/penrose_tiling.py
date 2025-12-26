import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

tau = np.pi * 2
D = 5          # number of directions
L = 31         # lines per direction
G = 0.25       # spacing

es = np.ones(D) / D   # same as JS default
off_x = 0.0
off_y = 0.0

def get_e_ang(e_i):
    return tau/4 + e_i * tau / D  # π/2 + 2πk/D

def line_intersection(p1, d1, p2, d2):
    """Intersection of two lines p1 + t d1 and p2 + s d2."""
    A = np.array([d1, -d2]).T
    b = p2 - p1
    try:
        t, s = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return p1 + t * d1

def compute_goal(pos, d1, d2, c):
    """CPU version of the shader’s 'goal' computation."""
    e_scale = 2.0 / D
    goal = np.zeros(2)
    for i in range(D):
        ang = get_e_ang(i)
        unit = np.array([np.cos(ang), np.sin(ang)])
        units = np.dot(unit, pos) / G - es[i]

        if i == d1:
            units = np.round(units)
            if c in (0, 1):
                units -= 1.0
        elif i == d2:
            units = np.round(units)
            if c in (1, 2):
                units -= 1.0
        else:
            units = np.floor(units)

        goal += unit * (units * G * e_scale)

    # offset (kept zero here, but you can add scroll logic)
    goal[0] += off_x
    goal[1] += off_y
    return goal

polys = []
colors = []

for d1 in range(D):
    for d2 in range(d1 + 1, D):
        ang1 = get_e_ang(d1)
        ang2 = get_e_ang(d2)
        unit1 = np.array([np.cos(ang1), np.sin(ang1)])
        unit2 = np.array([np.cos(ang2), np.sin(ang2)])
        side1 = np.array([np.cos(ang1 + tau/4), np.sin(ang1 + tau/4)])
        side2 = np.array([np.cos(ang2 + tau/4), np.sin(ang2 + tau/4)])

        for i in range(L):
            for j in range(L):
                len1 = (es[d1] + (i - L//2)) * G
                len2 = (es[d2] + (j - L//2)) * G
                p1 = unit1 * len1
                p2 = unit2 * len2
                pos = line_intersection(p1, side1, p2, side2)
                if pos is None:
                    continue

                # four vertices (c = 0,1,2,3)
                verts = [compute_goal(pos, d1, d2, c) for c in (0, 1, 2, 3)]
                v0, v1, v2, v3 = verts

                # two triangles per rhombus
                polys.append([v0, v1, v3])
                polys.append([v1, v2, v3])

                # color by angle difference
                diff = (d1 - d2) % D
                diff = min(diff, D - diff) - 1   # 0 or 1 for D=5
                if diff == 0:
                    col = (0.40, 0.65, 1.00, 1.0)  # blue-ish
                else:
                    col = (0.30, 0.80, 0.40, 1.0)  # green-ish
                colors.append(col)
                colors.append(col)

# ---- Plot ----
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')
fig.patch.set_facecolor("#111111")
ax.set_facecolor("#111111")

pc = PolyCollection(polys, facecolors=colors, edgecolors='none')
ax.add_collection(pc)

all_pts = np.vstack([np.array(p) for poly in polys for p in poly])
xmin, ymin = all_pts.min(axis=0)
xmax, ymax = all_pts.max(axis=0)
mx, my = 0.05 * (xmax - xmin), 0.05 * (ymax - ymin)
ax.set_xlim(xmin - mx, xmax + mx)
ax.set_ylim(ymin - my, ymax + my)
ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout()
plt.show()