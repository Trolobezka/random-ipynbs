# type: ignore

# https://openseespydoc.readthedocs.io/en/stable/src/ops_vis_ex_3d_3el_cantilever.html
import openseespy.opensees as ops
import openseespy.postprocessing.ops_vis as opsv
import matplotlib.pyplot as plt
import numpy as np

ops.wipe()
ops.model("basic", "-ndm", 3, "-ndf", 6)

# Dimensions and cross sections

d = 0.036  # m
R = 0.480  # m
L = 0.720  # m
q = 1875  # N/m

A = (1 / 2) * np.pi * d**2
I = (1 / 64) * np.pi * d**4
J = (1 / 32) * np.pi * d**4

E = 2e11  # Pa
G = 8e10  # Pa

# Define nodes

nodes = {}
n_R = 20
n_L = 10

F_total = 0.5 * q * L

node_index = 0
prev_load = 0
for angle in np.linspace(np.pi / 2, 0, n_R):
    nodes[node_index] = {"x": R * (1 - np.cos(angle)), "y": R * np.sin(angle), "z": 0}
    node_index += 1
for z in np.linspace(0, L, n_L + 1):
    if z == 0:
        continue
    nodes[node_index] = {"x": 0, "y": 0, "z": z, "load": z / L}
    node_index += 1

# Force correction
F_sum = sum([node.get("load", 0) for node in nodes.values()])
print("SUM before correction: " + str(F_sum))
for i, node in nodes.items():
    if "load" in node:
        node["load"] = F_total * node["load"] / F_sum
F_sum = sum([node.get("load", 0) for node in nodes.values()])
print("SUM after correction: " + str(F_sum))

# Create nodes
# ops.node(index, x, y, z)

for i, node in nodes.items():
    ops.node(i, nodes[i]["x"], nodes[i]["z"], nodes[i]["y"])

# Define constraints
# ops.fix(node_index, constraints(x, y, z, a, b, c))

ops.fix(0, 1, 1, 1, 1, 1, 1)

# Set node masses
# ops.mass(node_index, mass(x, y, z, a, b, c))

for i, node in nodes.items():
    ops.mass(i, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001)

# Transform stiffness from local to global space
# ops.geomTransf(type, index, dir_vec(x, y, z))
# https://openseespydoc.readthedocs.io/en/stable/src/LinearTransf.html

gT = 0
ops.geomTransf("Linear", gT, -1, 0, 0)

# Create beam segments

for i, node in nodes.items():
    node_1 = i
    node_2 = i + 1
    if node_2 in nodes:
        ops.element("elasticBeamColumn", i, node_1, node_2, A, E, G, J, I, I, gT)

# Create load

time_series_index = 1
ops.timeSeries("Constant", time_series_index)

pattern_index = 1
ops.pattern("Plain", pattern_index, time_series_index)

for i, node in nodes.items():
    if "load" in node:
        ops.load(i, -node["load"], 0, 0, 0, 0, 0)

# Setup simulation

ops.constraints("Transformation")
ops.numberer("RCM")
ops.system("BandGeneral")
ops.test("NormDispIncr", 1.0e-6, 6, 2)
ops.algorithm("Linear")
ops.integrator("LoadControl", 1)
ops.analysis("Static")
ops.analyze(1)

# Plot model

opsv.plot_model()
plt.title("Výpočtový model")
plt.gca().invert_yaxis()

# Plot deformation

scale_factor = 10
opsv.plot_defo(scale_factor)
plt.title("Deformace")
plt.gca().invert_yaxis()

# Plot extruded model

ele_shapes = {}
for i, node in nodes.items():
    ele_shapes[i] = ["circ", [d]]
opsv.plot_extruded_shapes_3d(ele_shapes)
plt.title("Grafický model")
plt.gca().invert_yaxis()

plt.show()
exit()
