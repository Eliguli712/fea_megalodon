import sys, subprocess
from pathlib import Path

# deps
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "numpy", "pydicom", "SimpleITK", "scikit-image", "scipy", "meshio", "matplotlib"],
               check=True)

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from skimage import filters, morphology, measure
import meshio
import matplotlib.pyplot as plt

import zipfile, shutil


# ----------------------------
# 1) Read DICOM series -> volume (z,y,x) + spacing (x,y,z)
#    UPDATED: folder may contain a single ZIP that holds the DICOM series.
# ----------------------------
def read_dicom_series(folder: Path):
    folder = Path(folder)
    assert folder.is_dir(), f"Not a folder: {folder}"

    # Case A) folder directly contains DICOM files -> read normally
    # Case B) folder contains exactly one zip -> extract to temp and read
    zips = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".zip"])
    has_dcm = any(p.is_file() and p.suffix.lower() in [".dcm", ".dicom"] for p in folder.iterdir())

    dicom_root = folder
    tmp_dir = None

    if (not has_dcm) and len(zips) == 1:
        # Extract ZIP
        zip_path = zips[0]
        tmp_dir = folder / "_unzipped_dicom_tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)

        # Sometimes ZIP contains another folder layer; choose the deepest folder that has DICOMs
        # Heuristic: find folder with most files
        candidate_dirs = [tmp_dir] + [p for p in tmp_dir.rglob("*") if p.is_dir()]
        best = None
        for d in candidate_dirs:
            files = list(d.glob("**/*"))
            # count likely dicom-ish files (including no extension)
            score = 0
            for f in files:
                if not f.is_file():
                    continue
                suf = f.suffix.lower()
                if suf in [".dcm", ".dicom"]:
                    score += 5
                elif suf in [".ima", ".img"]:
                    score += 2
                elif suf == "" and f.stat().st_size > 1024:
                    score += 1
            if best is None or score > best[0]:
                best = (score, d)
        dicom_root = best[1] if best is not None else tmp_dir

    # Now read with SimpleITK
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_root))
    if not series_ids:
        # If we extracted, clean up temp to avoid confusion
        raise RuntimeError(f"No DICOM series found. Looked in: {dicom_root}")

    files = reader.GetGDCMSeriesFileNames(str(dicom_root), series_ids[0])
    reader.SetFileNames(files)
    img = reader.Execute()  # SimpleITK Image

    vol_zyx = sitk.GetArrayFromImage(img).astype(np.float32)  # z,y,x
    spacing_xyz = img.GetSpacing()  # (x,y,z) in mm
    origin = img.GetOrigin()
    direction = img.GetDirection()

    meta = {
        "dicom_root_used": str(dicom_root.resolve()),
        "zip_extracted_from": str(zips[0].resolve()) if (not has_dcm and len(zips) == 1) else None,
        "series_ids_found": list(series_ids),
        "files_used": len(files),
    }
    return vol_zyx, spacing_xyz, origin, direction, files, meta


# ----------------------------
# 2) Force slice spacing to 16 µm if needed
# ----------------------------
def enforce_spacing_16um(spacing_xyz, z_um=16.0):
    sx, sy, _sz = map(float, spacing_xyz)
    sz2 = z_um / 1000.0  # mm
    return (sx, sy, sz2)


# ----------------------------
# 3) Save NIfTI intensity volume
# ----------------------------
def save_nifti(vol_zyx, spacing_xyz, out_path):
    img = sitk.GetImageFromArray(vol_zyx)  # z,y,x
    img.SetSpacing(tuple(map(float, spacing_xyz)))  # x,y,z
    sitk.WriteImage(img, str(out_path))


# ----------------------------
# 4) Rough tooth mask (threshold + morphology)
# ----------------------------
def rough_mask(vol_zyx):
    v = vol_zyx[np.isfinite(vol_zyx)]
    lo, hi = np.percentile(v, [5, 99.5])
    vv = np.clip(vol_zyx, lo, hi)

    thr = filters.threshold_otsu(vv)
    m = vv > thr

    m = morphology.remove_small_objects(m, 5000)
    m = morphology.binary_closing(m, morphology.ball(2))
    m = ndi.binary_fill_holes(m)

    lab, ncc = ndi.label(m)
    if ncc > 1:
        sizes = ndi.sum(m, lab, index=np.arange(1, ncc + 1))
        keep = 1 + int(np.argmax(sizes))
        m = (lab == keep)
    return m


# ----------------------------
# 5) Marching cubes -> (V,F) in mm coords
# ----------------------------
def mask_to_surface(mask_zyx, spacing_xyz):
    sx, sy, sz = map(float, spacing_xyz)  # x,y,z (mm)
    verts_zyx, faces, _, _ = measure.marching_cubes(
        mask_zyx.astype(np.uint8),
        level=0.5,
        spacing=(sz, sy, sx)  # (z,y,x) mm
    )
    V = np.column_stack([verts_zyx[:, 2], verts_zyx[:, 1], verts_zyx[:, 0]]).astype(np.float64)
    F = faces.astype(np.int32)
    return V, F


# ----------------------------
# 6) Export OBJ / STL / MSH
# ----------------------------
def write_obj(path, V, F):
    with open(path, "w") as f:
        f.write("# surface extracted from DICOM mask\n")
        for x, y, z in V:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in (F + 1):
            f.write(f"f {a} {b} {c}\n")

def write_stl(path, V, F):
    with open(path, "w") as f:
        f.write("solid tooth\n")
        for tri in F:
            p = V[tri]
            n = np.cross(p[1] - p[0], p[2] - p[0])
            nn = np.linalg.norm(n)
            if nn > 0:
                n = n / nn
            f.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n")
            f.write("    outer loop\n")
            for vv in p:
                f.write(f"      vertex {vv[0]:.6e} {vv[1]:.6e} {vv[2]:.6e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write("endsolid tooth\n")

def write_msh(path, V, F):
    meshio.write(path, meshio.Mesh(points=V, cells=[("triangle", F)]), file_format="gmsh")


# ----------------------------
# 7) Quick montage preview
# ----------------------------
def save_montage(vol_zyx, out_png, n=36):
    z = vol_zyx.shape[0]
    idx = np.linspace(0, z - 1, n).astype(int)

    p1, p99 = np.percentile(vol_zyx, [1, 99])
    vol01 = np.clip((vol_zyx - p1) / (p99 - p1 + 1e-12), 0, 1)

    cols = 6
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(12, 2.2 * rows), dpi=160)
    axs = np.array(axs).reshape(rows, cols)

    for i in range(rows * cols):
        ax = axs[i // cols, i % cols]
        ax.axis("off")
        if i < n:
            k = idx[i]
            ax.imshow(vol01[k], cmap="gray")
            ax.set_title(f"z={k}", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main(dicom_folder: str):
    dicom_folder = Path(dicom_folder)
    out_dir = Path(".")
    out_dir.mkdir(exist_ok=True)

    vol, spacing_xyz, origin, direction, files, meta = read_dicom_series(dicom_folder)

    # enforce 16 µm in z
    spacing_xyz = enforce_spacing_16um(spacing_xyz, z_um=16.0)

    nii_path = out_dir / "ct_16um.nii.gz"
    montage  = out_dir / "ct_preview_montage.png"
    obj_path = out_dir / "tooth_fullres.obj"
    stl_path = out_dir / "tooth_fullres.stl"
    msh_path = out_dir / "tooth_surface.msh"

    save_nifti(vol, spacing_xyz, nii_path)
    save_montage(vol, montage, n=36)

    mask = rough_mask(vol)
    V, F = mask_to_surface(mask, spacing_xyz)

    write_obj(obj_path, V, F)
    write_stl(stl_path, V, F)
    write_msh(msh_path, V, F)

    print({
        "dicom_folder": str(dicom_folder.resolve()),
        "meta": meta,
        "dicom_files_used": len(files),
        "volume_shape_zyx": tuple(vol.shape),
        "spacing_xyz_mm": spacing_xyz,
        "nii": str(nii_path.resolve()),
        "montage": str(montage.resolve()),
        "obj": str(obj_path.resolve()),
        "stl": str(stl_path.resolve()),
        "msh": str(msh_path.resolve()),
        "vertices": int(V.shape[0]),
        "faces": int(F.shape[0]),
    })


if __name__ == "__main__":
    # Jupyter 会注入 --f=... 这种参数，先过滤掉
    args = [a for a in sys.argv[1:] if not a.startswith("--f=")]

    # 允许：python script.py DICOM_16um
    # 或：不传参时默认用 ./DICOM_16um
    if len(args) >= 1:
        dicom_folder = args[0]
    else:
        dicom_folder = "DICOM_16um"

    main(dicom_folder)