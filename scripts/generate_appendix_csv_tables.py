from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "CT552_TraMy_Nguyen"
SOURCE_GROUPS = {
    "baseline_kfold_model_results": sorted(
        (ROOT / "outputs_LV" / "baseline_kfold_model_results").glob("k*.csv")
    ),
    "baseline_split_model_results": sorted(
        (ROOT / "outputs_LV" / "baseline_split_model_results").glob("k*.csv")
    ),
    "rules_extraction_results": sorted(
        (ROOT / "outputs_LV" / "rules_extraction_results").glob("k*.csv")
    ),
}
OUTPUT = PROJECT / "appendix-data" / "generated"


def tex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def make_table(source: Path, group: str) -> str:
    with source.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    header, data = rows[0], rows[1:]
    columns = [tex(item) for item in header]
    body = [
        " & ".join(tex(item) for item in row) + r"\\"
        for row in data
    ]
    caption = (
        f"Toàn bộ bản ghi trong {group}/{source.name}"
    )
    lines = [
        r"\begin{landscape}",
        r"\begingroup",
        r"\fontsize{7}{8}\selectfont",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{p{3.5cm}p{1.8cm}p{2.0cm}rrrrr}",
        rf"\caption{{{tex(caption)}}}\\",
        r"\hline",
        " & ".join(rf"\textbf{{{item}}}" for item in columns) + r"\\",
        r"\hline",
        r"\endfirsthead",
        r"\hline",
        " & ".join(rf"\textbf{{{item}}}" for item in columns) + r"\\",
        r"\hline",
        r"\endhead",
        *body,
        r"\hline",
        r"\end{longtable}",
        r"\endgroup",
        r"\end{landscape}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for group, files in SOURCE_GROUPS.items():
        for source in files:
            target = OUTPUT / f"{group}_{source.stem}.tex"
            target.write_text(
                make_table(source, group),
                encoding="utf-8",
                newline="\n",
            )


if __name__ == "__main__":
    main()
