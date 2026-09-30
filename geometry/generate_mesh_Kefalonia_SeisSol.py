import gmsh
import math
import shapely
from shapely.ops import transform
import pyproj

# ==========================================
# 1. GEOMETRIC & MESH PARAMETERS (in m)
# ==========================================
ZMIN = -300.0e3                # Bottom depth of the box (-300 km)
ZMAX = 0.0                     # Top surface of the box (0 km)
BOX_MARGIN = 300.0e3           # 300 km horizontal margin around the faults

FAULT_MESH_SIZE = 1.5e3        # Enforced element size directly on the faults (5 km)
MAX_MESH_SIZE = 50.0e3         # Maximum element size at domain boundaries (25 km)

# ==========================================
# 2. PROJECTION CONFIGURATION 
# ==========================================
proj_crs = pyproj.CRS("+proj=tmerc +datum=WGS84 +k=0.9996 +lon_0=20.8 +lat_0=38.50")
wgs84_crs = pyproj.CRS("EPSG:4326")
project_func = pyproj.Transformer.from_crs(wgs84_crs, proj_crs, always_xy=True).transform

fault_data = {
    "SouthernThrust": {"dip": 30, "dip_azimuth": 20, "extrusion_depth": -15e3, "wkt": "MULTILINESTRING ((20.7593619456553 38.0734467925457,20.644847423924 38.1203977464555,20.5211717404543 38.1719292812345,20.4478824465463 38.2417831394906,20.4249795422001 38.2807180768792,20.4055120735058 38.3414107733968,20.4020766378538 38.3654588229603))"},
    "NorthernThurst": {"dip": 30, "dip_azimuth": 20, "extrusion_depth": -15e3, "wkt": "MULTILINESTRING ((20.6741097089051 38.2870174848098,20.6278763725339 38.3010885002271,20.5766176735136 38.3211899508233,20.5384249173809 38.3553624168368,20.5183234667847 38.385514592731,20.4911865084799 38.4417986544003,20.4720901304135 38.4880319907716))"},
    "KefaloniaTransformFault": {"dip": 80, "dip_azimuth": 90, "extrusion_depth": -30e3, "wkt": "MULTILINESTRING ((20.6724776088622 38.9530444872599,20.6513332812345 38.8857670811719,20.6128890491842 38.7800454430336,20.5571449127113 38.6435684192551,20.5302339502761 38.5724465899621,20.5033229878409 38.493635914259,20.4571899093806 38.3821476413132,20.4129790425228 38.284114849585,20.3899125032926 38.2514372523422,20.3207128856021 38.1745487882417,20.2861130767568 38.0957381125386,20.2495910563091 37.9650277235676,20.1861580734261 37.8881392594671,20.1515582645808 37.8343173345967))"}
}


# ==========================================
# 3. GMSH INITIALIZATION & SURFACE CREATION
# ==========================================
gmsh.initialize()
gmsh.model.add("fault_system")
occ = gmsh.model.occ

all_x = []
all_y = []

# Store surfaces by name
fault_surfaces = {}

for fid, info in fault_data.items():
    geom = shapely.from_wkt(info["wkt"])
    projected_geom = transform(project_func, geom)
    coords = list(projected_geom.geoms[0].coords)

    # Calculate structural dip offset vector component
    dip_rad = math.radians(info["dip"])
    horizontal_offset = abs(info["extrusion_depth"]) / math.tan(dip_rad)

    # Calculate horizontal offset vector based on azimuth (0 deg = North, 90 deg = East)
    azimuth_rad = math.radians(info["dip_azimuth"])
    dx = horizontal_offset * math.sin(azimuth_rad)
    dy = horizontal_offset * math.cos(azimuth_rad)

    # Track boundary points for the volume calculation
    for pt in coords:
        all_x.extend([pt[0], pt[0] + dx])
        all_y.extend([pt[1], pt[1] + dy])

    # Track boundary points for the volume calculation
    for pt in coords:
        all_x.extend([pt[0], pt[0] + dx])
        all_y.extend([pt[1], pt[1] + dy])

    # Instantiate Gmsh points
    top_points = []
    bottom_points = []
    for pt in coords:
        top_points.append(occ.addPoint(pt[0], pt[1], 0.0))
        bottom_points.append(occ.addPoint(pt[0] + dx, pt[1] + dy, info["extrusion_depth"]))

    # Generate smooth trace curves using OpenCASCADE BSplines
    top_spline = occ.addBSpline(top_points)
    bottom_spline = occ.addBSpline(bottom_points)

    # Generate lateral edges
    left_bound = occ.addLine(top_points[0], bottom_points[0])
    right_bound = occ.addLine(top_points[-1], bottom_points[-1])

    # Close into a boundary wire loop
    wire_loop = occ.addCurveLoop([top_spline, right_bound, -bottom_spline, -left_bound])

    # Interpolate smooth surface
    fault_surface = occ.addSurfaceFilling(wire_loop)
    fault_surfaces[fid] = (2, fault_surface)

# Synchronize OCC so geometries are registered
occ.synchronize()

# ==========================================
# 4. SLICE & CLEAN THRUST FAULTS
# ==========================================
transform_tag = fault_surfaces["KefaloniaTransformFault"]
thrust_tags = [fault_surfaces["SouthernThrust"], fault_surfaces["NorthernThurst"]]

# Fragment all fault surfaces together to slice thrusts at the transform boundary
all_fault_tags = [transform_tag] + thrust_tags
out_dim_tags, out_dim_tags_map = occ.fragment(all_fault_tags, [])
occ.synchronize()
print(out_dim_tags_map)

# Identify resulting surfaces for each original thrust fault and discard the smaller piece
final_surface_tags = []

# Keep the transform fault surface(s)
transform_map_idx = 0
for dim_tag in out_dim_tags_map[transform_map_idx]:
    if dim_tag[0] == 2:  # Surface dimension
        final_surface_tags.append(dim_tag)

# Filter each thrust fault
for idx, thrust_name in enumerate(["SouthernThrust", "NorthernThurst"], start=1):
    mapped_surfaces = [dt for dt in out_dim_tags_map[idx] if dt[0] == 2]

    if len(mapped_surfaces) > 1:
        # Get the X-coordinate of the Center of Mass for each sub-surface piece
        # occ.getCenterOfMass returns (x, y, z)
        com_x_list = [(dt, occ.getCenterOfMass(dt[0], dt[1])[0]) for dt in mapped_surfaces]

        # Sort by X-coordinate in ascending order (most negative / furthest West first)
        com_x_list.sort(key=lambda item: item[1])

        # Remove the westernmost surface tag (the first item after sorting)
        westernmost_surface = com_x_list[0][0]
        occ.remove([westernmost_surface], recursive=True)

        # Retain the remaining eastern piece(s)
        for dt, _ in com_x_list[1:]:
            final_surface_tags.append(dt)
    else:
        final_surface_tags.extend(mapped_surfaces)

occ.synchronize()

# ==========================================
# 5. DOMAIN BOX & VOLUME FRAGMENTATION
# ==========================================
xmin, xmax = min(all_x) - BOX_MARGIN, max(all_x) + BOX_MARGIN
ymin, ymax = min(all_y) - BOX_MARGIN, max(all_y) + BOX_MARGIN

dx_box = xmax - xmin
dy_box = ymax - ymin
dz_box = ZMAX - ZMIN

# Build the bounding volume starting from the deep corner
box_tag = occ.addBox(xmin, ymin, ZMIN, dx_box, dy_box, dz_box)

print("Slicing 3D volume blocks using OpenCASCADE...")
fused, fused_map = occ.fragment([(3, box_tag)], final_surface_tags)
occ.synchronize()

# --- A. TAG FAULT SURFACES (165, 166, 167) ---

# Map fault names to physical group IDs
fault_physical_ids = {
    "KefaloniaTransformFault": 3,
    "SouthernThrust": 65,
    "NorthernThurst": 66
}

# Dictionary to accumulate all resulting surface tags for each fault
fault_surface_collector = {
    "KefaloniaTransformFault": [],
    "SouthernThrust": [],
    "NorthernThurst": []
}

for idx, surface_dt in enumerate(final_surface_tags):
    # Determine parent fault name
    original_name = None
    if surface_dt in out_dim_tags_map[0]:
        original_name = "KefaloniaTransformFault"
    elif surface_dt in out_dim_tags_map[1]:
        original_name = "SouthernThrust"
    elif surface_dt in out_dim_tags_map[2]:
        original_name = "NorthernThurst"

    # Retrieve all sub-surfaces generated after 3D fragmenting (fused_map index shift +1)
    resulting_surfaces = [dt[1] for dt in fused_map[idx + 1] if dt[0] == 2]

    if original_name and resulting_surfaces:
        fault_surface_collector[original_name].extend(resulting_surfaces)

# Now assign each physical group once with all accumulated surface tags
for name, phys_id in fault_physical_ids.items():
    tags = fault_surface_collector[name]
    if tags:
        gmsh.model.addPhysicalGroup(2, tags, tag=phys_id, name=name)


# --- B. TAG BOX EXTERNAL (5) & TOP (1) MUTUALLY EXCLUSIVELY ---
top_box_surfaces = []
external_box_surfaces = []

# Get all 2D surfaces in the model after slicing
all_surfaces = gmsh.model.getEntities(dim=2)

for dim, s_tag in all_surfaces:
    # Get center of mass for each surface to test its spatial position
    com = occ.getCenterOfMass(dim, s_tag)
    x, y, z = com[0], com[1], com[2]

    # Check if surface sits on the top boundary (ZMAX = 0.0)
    if math.isclose(z, ZMAX, abs_tol=1e-3):
        top_box_surfaces.append(s_tag)
    # Check lateral and bottom boundaries only (excluding the top)
    elif (math.isclose(x, xmin, abs_tol=1e-3) or math.isclose(x, xmax, abs_tol=1e-3) or
          math.isclose(y, ymin, abs_tol=1e-3) or math.isclose(y, ymax, abs_tol=1e-3) or
          math.isclose(z, ZMIN, abs_tol=1e-3)):
        external_box_surfaces.append(s_tag)

# Create mutually exclusive Physical Groups
if top_box_surfaces:
    gmsh.model.addPhysicalGroup(2, top_box_surfaces, tag=1, name="Top_Surface")

if external_box_surfaces:
    gmsh.model.addPhysicalGroup(2, external_box_surfaces, tag=5, name="External_Boundaries")

# --- C. TAG ALL 3D VOLUME REGIONS (Tag 1) ---
# Retrieve all 3D volume entities generated after OpenCASCADE fragmentation
volume_entities = [v[1] for v in gmsh.model.getEntities(dim=3)]

if volume_entities:
    # Assign all 3D volume tags to Physical Volume 1
    gmsh.model.addPhysicalGroup(3, volume_entities, tag=1, name="Domain_Volume")


gmsh.model.occ.synchronize()

# ==========================================
# 6. MESH SIZE FIELDS
# ==========================================
# Clear strict default constraints to allow Mesh Fields to take absolute control
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)

# Isolate the internal fault surfaces resulting from the fragmentation
all_surfaces = gmsh.model.getEntities(2)
fault_surface_ids = []

eps = 0.1
for dim, tag in all_surfaces:
    bbox = gmsh.model.getBoundingBox(dim, tag)
    # If the surface boundaries fall inside the outer box limits, it belongs to a fault plane
    if (bbox[0] > xmin + eps and bbox[3] < xmax - eps and
        bbox[1] > ymin + eps and bbox[4] < ymax - eps):
        fault_surface_ids.append(tag)


# Field 1: Distance tracking from internal fault surface patches
field_dist = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(field_dist, "SurfacesList", fault_surface_ids)
gmsh.model.mesh.field.setNumber(field_dist, "Sampling", 100)

# Field 2: Continuous MathEval equation establishing the target growth rate.
# F{field_dist} represents the raw perpendicular distance from the fault plane.
# As distance increases, the mesh size increases linearly at a 1:1 rate.
field_math = gmsh.model.mesh.field.add("MathEval")
gmsh.model.mesh.field.setString(field_math, "F", f"F{field_dist} + {FAULT_MESH_SIZE}")

# Field 3: Max constraint field. This acts as our ceiling cap.
field_max_cap = gmsh.model.mesh.field.add("MathEval")
gmsh.model.mesh.field.setString(field_max_cap, "F", f"{MAX_MESH_SIZE}")

# Field 4: Use a Min field to choose the smaller value between the growing slope and the ceiling cap.
# This cleanly handles the transition: close to faults it uses Field 2, far away it uses Field 3.
field_final = gmsh.model.mesh.field.add("Min")
gmsh.model.mesh.field.setNumbers(field_final, "FieldsList", [field_math, field_max_cap])

# Set this sizing profile as the active global background mesh field
gmsh.model.mesh.field.setAsBackgroundMesh(field_final)

# Frontal-Delaunay 3D handles variable-scale volume grading transitions effectively
gmsh.option.setNumber("Mesh.Algorithm3D", 1) 


# Configure Gmsh to ONLY save elements belonging to Physical Groups
gmsh.option.setNumber("Mesh.SaveAll", 0)

# Generate complete 3D volume mesh
print("Generating 3D volume mesh...")
gmsh.model.mesh.generate(3)

# Force Gmsh to export in MSH 2.2 format
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

# Save final physical MSH file
output_file = "Kefalonia_3faults_300km_box.msh"
gmsh.write(output_file)
print(f"3D mesh successfully exported to {output_file}")



# Launch GUI viewer for inspection
gmsh.fltk.run()
gmsh.finalize()

