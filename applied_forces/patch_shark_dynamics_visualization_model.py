#!/usr/bin/env python3

import argparse
import io
import re
import shutil
import zipfile
from pathlib import Path


VIEW_CAMERA = {
    "view2": {
        "orthoscale": "39.18271156946385",
        "zoomanglefull": "32",
        "position": "1|3,'13.778231603838151','19.44345875793703','69.63722487812342'",
        "target": "1|3,'15.848614932448989','17.055069531397102','44.837847672734604'",
        "up": "1|3,'0.007948164750532853','0.9954260168611613','-0.09520436772029735'",
        "rotationpoint": "1|3,'15.848614932448989','17.055069531397102','44.837847672734604'",
        "viewoffset": "1|2,'0','0'",
    },
    "view3": {
        "orthoscale": "39.18271156946385",
        "zoomanglefull": "32",
        "position": "1|3,'13.778231603838151','19.44345875793703','69.63722487812342'",
        "target": "1|3,'15.848614932448989','17.055069531397102','44.837847672734604'",
        "up": "1|3,'0.007948164750532853','0.9954260168611613','-0.09520436772029735'",
        "rotationpoint": "1|3,'15.848614932448989','17.055069531397102','44.837847672734604'",
        "viewoffset": "1|2,'0','0'",
    },
}

PLOT_TAGS = (
    "pg_holo_std1",
    "pg_holo_std_nh",
    "pg_holo_std_og",
    "pg_holo_std_mr2",
    "pg_holo_std_mr5",
    "pg_vms_holocastic",
)

HYPER_TAGS = ("hmm_nh", "hmm_og", "hmm_mr2", "hmm_mr5")


def remove_hyperelastic_blocks(text: str) -> str:
    for tag in HYPER_TAGS:
        pattern = re.compile(
            rf'<PhysicsFeature op="HyperelasticModel" tag="{tag}".*?</PhysicsFeature>\n?',
            re.S,
        )
        text, count = pattern.subn("", text)
        if count == 0:
            raise RuntimeError(f"Missing hyperelastic block for {tag}")
    return text


def _replace_property(block: str, name: str, value: str, required: bool = True) -> str:
    full_pattern = re.compile(
        rf'(<propertyValue T="30"[^>]*name="{re.escape(name)}"[^>]*>)',
        re.S,
    )
    match = full_pattern.search(block)
    if not match:
        if required:
            raise RuntimeError(f"Missing property {name}")
        return block

    prop = match.group(1)
    prop_new, count = re.subn(r'(value(?:Matrix)?=")([^"]*)(")', rf"\g<1>{value}\3", prop, count=1)
    if count == 0 and required:
        raise RuntimeError(f"Missing property {name}")
    return block[: match.start(1)] + prop_new + block[match.end(1) :]


def patch_view_block(text: str, view_tag: str, props: dict[str, str], required: bool = True) -> str:
    pattern = re.compile(
        rf'(<View op="ModelView3D" tag="{view_tag}".*?<ViewFeature op="Camera" tag="camera".*?>)(.*?)(</ViewFeature>)',
        re.S,
    )
    match = pattern.search(text)
    if not match:
        if not required:
            return text
        raise RuntimeError(f"Missing camera block for {view_tag}")
    camera_body = match.group(2)
    for name, value in props.items():
        camera_body = _replace_property(camera_body, f"p:{name}", value, required=required)
    return text[: match.start(2)] + camera_body + text[match.end(2) :]


def bind_plot_group_views(text: str, required: bool = True) -> str:
    for tag in PLOT_TAGS:
        pattern = re.compile(
            rf'(<ResultFeature op="PlotGroup3D" tag="{tag}".*?<propertyValue T="30" value=")auto(" name="p:view" ReferenceContainerList="/view" CompositeIndex="0"></propertyValue>)',
            re.S,
        )
        text, count = pattern.subn(rf"\1view3\2", text, count=1)
        if count == 0 and required:
            raise RuntimeError(f"Missing auto-view property for {tag}")
    return text


def patch_xml_text(text: str, nested: bool = False) -> str:
    try:
        text = remove_hyperelastic_blocks(text)
    except RuntimeError:
        if not nested:
            raise
    for view_tag, props in VIEW_CAMERA.items():
        text = patch_view_block(text, view_tag, props, required=not nested)
    text = bind_plot_group_views(text, required=not nested)
    return text


def patch_usedlicenses(data: bytes) -> bytes:
    lines = [line for line in data.decode("utf-8").splitlines() if line]
    lines = [line for line in lines if line not in {"CADIMPORT", "NONLINEARSTRUCTMATERIALS"}]
    return ("\n".join(lines) + "\n").encode("utf-8")


def patch_nested_model_zip(data: bytes) -> bytes:
    src = zipfile.ZipFile(io.BytesIO(data), "r")
    out_buf = io.BytesIO()
    with zipfile.ZipFile(out_buf, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as dst:
        for info in src.infolist():
            payload = src.read(info.filename)
            if info.filename == "dmodel.xml":
                payload = patch_xml_text(payload.decode("utf-8", errors="ignore"), nested=True).encode("utf-8")
            new_info = zipfile.ZipInfo(info.filename, date_time=info.date_time)
            new_info.compress_type = zipfile.ZIP_DEFLATED
            new_info.comment = info.comment
            new_info.extra = info.extra
            new_info.internal_attr = info.internal_attr
            new_info.external_attr = info.external_attr
            new_info.create_system = info.create_system
            dst.writestr(new_info, payload)
    src.close()
    return out_buf.getvalue()


def patch_archive(src_path: Path, dst_path: Path) -> None:
    with zipfile.ZipFile(src_path, "r") as src, zipfile.ZipFile(
        dst_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
    ) as dst:
        dst.comment = src.comment
        for info in src.infolist():
            payload = src.read(info.filename)
            if info.filename == "dmodel.xml":
                payload = patch_xml_text(payload.decode("utf-8", errors="ignore"), nested=False).encode("utf-8")
            elif info.filename == "usedlicenses.txt":
                payload = patch_usedlicenses(payload)
            elif info.filename.startswith("savepoint") and info.filename.endswith("model.zip"):
                payload = patch_nested_model_zip(payload)

            new_info = zipfile.ZipInfo(info.filename, date_time=info.date_time)
            new_info.compress_type = zipfile.ZIP_DEFLATED
            new_info.comment = info.comment
            new_info.extra = info.extra
            new_info.internal_attr = info.internal_attr
            new_info.external_attr = info.external_attr
            new_info.create_system = info.create_system
            dst.writestr(new_info, payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a visualization-safe shark dynamics MPH with front-facing head view."
    )
    parser.add_argument("--src", required=True, help="Source MPH file")
    parser.add_argument("--dst", required=True, help="Destination MPH file")
    parser.add_argument("--backup", help="Optional backup path for the destination before overwrite")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    tmp = dst.with_suffix(dst.suffix + ".tmp")

    if args.backup and dst.exists():
        shutil.copy2(dst, args.backup)

    patch_archive(src, tmp)
    tmp.replace(dst)


if __name__ == "__main__":
    main()
