"""Generate a MuJoCo XML from the simplified URDF."""

from __future__ import annotations

from pathlib import Path

from .mjcf_utils import write_mjcf


ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = ROOT / "assets" / "urdf" / "Lagrange_01_sim.urdf"
MESH_DIR = (ROOT / "assets" / "meshes").resolve()
OUTPUT_XML = ROOT / "assets" / "mjcf" / "lagrange_generated.xml"


def main():
    path = write_mjcf(URDF_PATH, MESH_DIR, OUTPUT_XML)
    print(f"Generated MJCF: {path}")


if __name__ == "__main__":
    main()
