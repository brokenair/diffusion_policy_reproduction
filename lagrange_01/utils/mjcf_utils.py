"""
Minimal URDF -> MJCF helper tailored to Lagrange_01.
Keeps mesh visuals, inertial data, and hinge/slide joints so MuJoCo can visualize
the arm without ROS.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]


def _parse_vec(text: Optional[str], length: int = 3) -> Vec3:
    if not text:
        return (0.0, 0.0, 0.0)
    parts = [float(x) for x in text.split()]
    if len(parts) != length:
        raise ValueError(f"Expected {length} values, got {text}")
    return tuple(parts)  # type: ignore


def rpy_to_quat(rpy: Vec3) -> Quat:
    roll, pitch, yaw = rpy
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (w, x, y, z)


def _origin(elem: Optional[ET.Element]) -> Tuple[Vec3, Quat]:
    if elem is None:
        return (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)
    xyz = _parse_vec(elem.attrib.get("xyz"))
    rpy = _parse_vec(elem.attrib.get("rpy"))
    return xyz, rpy_to_quat(rpy)


def _format(vec: Iterable[float]) -> str:
    return " ".join(f"{v:.8g}" for v in vec)


def load_urdf(urdf_path: Path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    materials: Dict[str, List[float]] = {}
    for m in root.findall("material"):
        name = m.attrib.get("name")
        color = m.find("color")
        if name and color is not None and color.attrib.get("rgba"):
            rgba = [float(x) for x in color.attrib["rgba"].split()]
            materials[name] = rgba

    links: Dict[str, Dict] = {}
    for link in root.findall("link"):
        name = link.attrib["name"]
        inertial_elem = link.find("inertial")
        inertial = None
        if inertial_elem is not None:
            pos, quat = _origin(inertial_elem.find("origin"))
            inertia_elem = inertial_elem.find("inertia")
            mass_elem = inertial_elem.find("mass")
            if inertia_elem is not None and mass_elem is not None:
                inertial = {
                    "pos": pos,
                    "quat": quat,
                    "mass": float(mass_elem.attrib["value"]),
                    "diaginertia": (
                        float(inertia_elem.attrib["ixx"]),
                        float(inertia_elem.attrib["iyy"]),
                        float(inertia_elem.attrib["izz"]),
                    ),
                }

        visuals = []
        for vis in link.findall("visual"):
            pos, quat = _origin(vis.find("origin"))
            geom = vis.find("geometry")
            material_elem = vis.find("material")
            material_name = None
            rgba = None
            if material_elem is not None:
                material_name = material_elem.attrib.get("name") or None
                color = material_elem.find("color")
                if color is not None and color.attrib.get("rgba"):
                    rgba = [float(x) for x in color.attrib["rgba"].split()]
            visuals.append(
                {
                    "origin": (pos, quat),
                    "geom": geom,
                    "material_name": material_name,
                    "rgba": rgba,
                }
            )

        links[name] = {"inertial": inertial, "visuals": visuals}

    joints: Dict[str, Dict] = {}
    for joint in root.findall("joint"):
        jtype = joint.attrib["type"]
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        pos, quat = _origin(joint.find("origin"))
        axis_elem = joint.find("axis")
        axis = _parse_vec(axis_elem.attrib["xyz"]) if axis_elem is not None else (0.0, 0.0, 1.0)
        limit_elem = joint.find("limit")
        limits = None
        if limit_elem is not None and "lower" in limit_elem.attrib and "upper" in limit_elem.attrib:
            limits = (float(limit_elem.attrib["lower"]), float(limit_elem.attrib["upper"]))
        joints[joint.attrib["name"]] = {
            "type": jtype,
            "parent": parent,
            "child": child,
            "origin": (pos, quat),
            "axis": axis,
            "limits": limits,
        }

    return links, joints, materials


def urdf_to_mjcf_xml(urdf_path: Path, mesh_dir: Path) -> str:
    links, joints, materials = load_urdf(urdf_path)
    child_map: Dict[str, List[str]] = defaultdict(list)
    for name, joint in joints.items():
        child_map[joint["parent"]].append(name)

    children = {joint["child"] for joint in joints.values()}
    roots = [l for l in links.keys() if l not in children]
    if not roots:
        raise RuntimeError("No root link found in URDF")
    if len(roots) > 1:
        raise RuntimeError(f"Multiple root links not supported: {roots}")
    root_link = roots[0]

    mesh_assets = set()
    for link in links.values():
        for vis in link["visuals"]:
            geom = vis["geom"]
            if geom is None:
                continue
            mesh_elem = geom.find("mesh") if geom is not None else None
            if mesh_elem is not None and mesh_elem.attrib.get("filename"):
                mesh_file = mesh_elem.attrib["filename"]
                mesh_name = Path(mesh_file).name
                mesh_assets.add(mesh_name)

    lines: List[str] = []
    lines.append(f'<mujoco model="{urdf_path.stem}">')
    lines.append(f'  <compiler angle="radian" meshdir="{mesh_dir}"/>')
    lines.append('  <option gravity="0 0 -9.81" timestep="0.002"/>')
    if mesh_assets:
        lines.append("  <asset>")
        for mesh in sorted(mesh_assets):
            lines.append(f'    <mesh name="{mesh}" file="{mesh}"/>')
        lines.append("  </asset>")

    def emit_inertial(inertial: Dict, indent: str):
        diaginertia = inertial["diaginertia"]
        lines.append(
            f'{indent}<inertial pos="{_format(inertial["pos"])}" '
            f'quat="{_format(inertial["quat"])}" '
            f'mass="{inertial["mass"]:.8g}" '
            f'diaginertia="{diaginertia[0]:.8g} {diaginertia[1]:.8g} {diaginertia[2]:.8g}"/>'
        )

    def emit_visuals(link_name: str, visuals: List[Dict], indent: str):
        for idx, vis in enumerate(visuals):
            geom = vis["geom"]
            if geom is None:
                continue
            mesh_elem = geom.find("mesh")
            box_elem = geom.find("box")
            pos, quat = vis["origin"]
            rgba = None
            if vis["rgba"]:
                rgba = vis["rgba"]
            elif vis["material_name"] and vis["material_name"] in materials:
                rgba = materials[vis["material_name"]]
            rgba_str = _format(rgba) if rgba else "0.7 0.7 0.7 1"
            if mesh_elem is not None and mesh_elem.attrib.get("filename"):
                mesh_name = Path(mesh_elem.attrib["filename"]).name
                lines.append(
                    f'{indent}<geom name="{link_name}_vis_{idx}" type="mesh" '
                    f'mesh="{mesh_name}" pos="{_format(pos)}" quat="{_format(quat)}" '
                    f'rgba="{rgba_str}" contype="1" conaffinity="1"/>'
                )
            elif box_elem is not None and box_elem.attrib.get("size"):
                size = box_elem.attrib["size"]
                lines.append(
                    f'{indent}<geom name="{link_name}_box_{idx}" type="box" '
                    f'size="{size}" pos="{_format(pos)}" quat="{_format(quat)}" '
                    f'rgba="{rgba_str}" contype="1" conaffinity="1"/>'
                )

    def emit_body(link_name: str, indent: str, joint_name: Optional[str]):
        joint = joints.get(joint_name) if joint_name else None
        pos, quat = (joint["origin"] if joint else ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)))
        lines.append(f'{indent}<body name="{link_name}" pos="{_format(pos)}" quat="{_format(quat)}">')
        if joint and joint["type"] != "fixed":
            jtype = "hinge" if joint["type"] in ("revolute", "continuous") else "slide"
            limited = "false" if joint["type"] == "continuous" or joint["limits"] is None else "true"
            range_attr = ""
            if joint["limits"] is not None:
                range_attr = f' range="{joint["limits"][0]:.8g} {joint["limits"][1]:.8g}"'
            lines.append(
                f'{indent}  <joint name="{joint_name}" type="{jtype}" axis="{_format(joint["axis"])}"'
                f' limited="{limited}"{range_attr} damping="0.1"/>'
            )
        inertial = links[link_name]["inertial"]
        if inertial:
            emit_inertial(inertial, indent + "  ")
        emit_visuals(link_name, links[link_name]["visuals"], indent + "  ")
        for child_joint in child_map.get(link_name, []):
            emit_body(joints[child_joint]["child"], indent + "  ", child_joint)
        lines.append(f"{indent}</body>")

    lines.append("  <worldbody>")
    emit_body(root_link, "    ", None)
    lines.append("  </worldbody>")
    
    # Add actuators for manual control in MuJoCo viewer
    actuated_joints = [
        name for name, joint in joints.items() 
        if joint["type"] in ("revolute", "continuous", "prismatic")
    ]
    if actuated_joints:
        lines.append("  <actuator>")
        for joint_name in sorted(actuated_joints):
            # Position servo actuator for manual control
            lines.append(f'    <position name="{joint_name}_actuator" joint="{joint_name}" '
                        f'kp="100" ctrlrange="-3.14 3.14"/>')
        lines.append("  </actuator>")
    
    lines.append("</mujoco>")
    return "\n".join(lines)


def write_mjcf(urdf_path: Path, mesh_dir: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    xml = urdf_to_mjcf_xml(urdf_path, mesh_dir)
    output_path.write_text(xml)
    return output_path
