# CV5570 – Construction-site worker risk scoring

Course project (CV5570, IIT Madras). Given a site photo, the pipeline finds workers, PPE and machines, estimates camera geometry / metric depth and body pose, and produces three rule-based scores per worker (PPE, struck-by, fall) plus one combined risk score and band.

```
image -> detector (Person / PPE / machines) -> pose (17 joints) -> GeoCalib + Depth Pro
      -> PPE engine + struck-by engine + fall engine (0-100 each)
      -> weighted aggregator (weights from BLS construction-fatality shares) -> risk + band
```

Nothing in the scoring layer is learned; every score can be traced back to its triggers.

## Repository layout

The folder structure mirrors the `cv5570_risk/` folder on Google Drive that the notebooks read and write, so you can clone this repo straight into Drive.

| Folder | Contents |
|---|---|
| `notebooks/` | The 8 Colab notebooks (outputs stripped) |
| `code/risk_pose.py` | Shared pose / elevation / posture module (written by NB6, imported by NB4 and NB8) |
| `weights/` | Fine-tuned detector (`det_ppe_worker_best.pt`), per-class thresholds, isotonic calibration, `risk_config.json`, before/after tables |
| `runs/` | Training curves, confusion matrices, `results.csv` for the detector runs |
| `diagnostics/` | NB7 outputs (recall by object size, machine sweep, PPE priors by head size) |
| `accuracy/` | NB5 distance-accuracy study (images, GeoCalib camera files, result tables and plots) |
| `demo/images/` | The 9 demo images |
| `data/` | Empty. Put the dataset zips here (not tracked, see below) |

## Notebooks (run order)

| # | Notebook | Purpose |
|---|---|---|
| 1 | `nb1_object_detector` | Fine-tune the 10-class YOLOv8n on a MOCS + CSS mix; fit thresholds and calibration |
| 7 | `nb7_diagnostics_calibration_v2` | Detector diagnostics; writes `weights/risk_config.json` |
| 2 | `nb2_depth_pro` | Apple Depth Pro metric depth -> `depth/<stem>.npz` |
| 3 | `nb3_geocalib_run` | GeoCalib camera + gravity -> `camera/<stem>.json` |
| 6 | `nb6_pose_estimator_v2` | Pose, posture, elevation; writes `code/risk_pose.py` |
| 4 | `nb4_engines_deployment_v3` | The three engines, aggregator, batch scoring and annotated outputs |
| 5 | `nb5_distance_accuracy` | Depth Pro vs GeoCalib vs tape-measure distance study |
| 8 | `nb8_live_demo_gradio` | Live Gradio demo (upload a photo, pick the distance method) |

All notebooks are written for **Google Colab (T4 GPU)** and expect `/content/drive/MyDrive/cv5570_risk` as the root.

## Reproducing

1. Copy (or clone) this repo into Google Drive as `MyDrive/cv5570_risk`.
2. Put the two dataset zips in `cv5570_risk/data/`:
   - `css_export.zip` – Construction Site Safety data plus the earlier `old_yolov8n_best.pt`
   - `MOCS_v1i_yolo26.zip` – MOCS export (worker boxes only)
3. Open the notebooks in Colab in the order above. Each one installs its own dependencies.

To run only the demo you do not need the datasets: the committed `weights/` and `code/risk_pose.py` are enough for NB8.

## Notes

- Notebook outputs were stripped to keep the repo small; re-run to regenerate them.
- Depth maps (`*.npz`), pose JSONs and result folders are regenerable and ignored by git.
- Aggregator weights: fall 0.46, PPE 0.29, struck-by 0.25 (+0.15 pairwise term), derived from BLS CFOI 2023 fatality shares. Band cut-offs are placeholders.
