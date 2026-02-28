## 16um CT Dataset

The raw archive for this dataset is `DICOM_16um/16um uCT Scanco.zip`.

It is not committed directly because the file is about 19 GB, which is too large for GitHub file storage limits. The repository instead includes:

- `DICOM_16um/16um_uCT_Scanco.zip.release.json`
- `scripts/publish_dicom_16um_release.py`
- `scripts/fetch_dicom_16um_release.py`

The manifest records the exact split layout and SHA-256 checksums for a GitHub Release upload under tag `ct-dicom-16um`.

Publish the split archive as release assets:

```bash
GITHUB_TOKEN=... python3 scripts/publish_dicom_16um_release.py
```

Fetch and reassemble the archive from release assets:

```bash
python3 scripts/fetch_dicom_16um_release.py
```
