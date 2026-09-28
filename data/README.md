# Datasets

Download and verify all four official files with:

```bash
python scripts/download_datasets.py
python scripts/download_datasets.py --verify-only
```

The files are stored under `data/raw/`. They are intentionally excluded from normal Git history: the four records use the **4TU General Terms of Use**, and keeping generated clones out of Git avoids repository bloat. `data/manifest.json` pins the official URLs, DOIs, byte sizes, and SHA-256 checksums, so every clone retrieves the same bytes.

| Folder | Expected file | Official source |
|---|---|---|
| `bpic2017/` | `BPI Challenge 2017.xes.gz` | https://data.4tu.nl/articles/dataset/BPI_Challenge_2017/12696884 |
| `bpic2012/` | `BPI Challenge 2012.xes.gz` | https://data.4tu.nl/articles/dataset/BPI_Challenge_2012/12689204 |
| `helpdesk/` | `finale.csv` | https://data.4tu.nl/articles/dataset/Dataset_belonging_to_the_help_desk_log_of_an_Italian_Company/12675977 |
| `sepsis/` | `Sepsis Cases - Event Log.xes.gz` | https://data.4tu.nl/articles/dataset/Sepsis_Cases_-_Event_Log/12707639 |

The Helpdesk dataset required by the paper is the 4,580-case, 14-activity 4TU version. Do not substitute the unrelated 3,804-case Helpdesk log.
