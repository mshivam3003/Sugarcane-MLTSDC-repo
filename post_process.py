
from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


try:
    from PIL import Image, ImageOps
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: Pillow. Install with: `pip install pillow`"
    ) from exc


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class PromptRow:
    id: int
    pair_code: str
    disease_pair: str
    severity: str
    climate: str
    folder: str


def _slugify(value: str) -> str:
    value = value.strip()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^A-Za-z0-9_.-]+", "", value)
    return value or "unknown"


def load_prompts(prompt_csv_path: Path) -> dict[int, PromptRow]:
    # Use utf-8-sig to transparently handle Excel/Sheets CSVs with a UTF-8 BOM.
    with prompt_csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = [fn.strip() for fn in (reader.fieldnames or []) if fn is not None]
        required = {"ID", "Pair_Code", "Disease_Pair", "Severity", "Climate", "Folder"}
        missing = required - set(fieldnames)
        if missing:
            raise ValueError(f"{prompt_csv_path} missing columns: {sorted(missing)}")

        rows: dict[int, PromptRow] = {}
        for row in reader:
            row = {str(k).strip(): v for k, v in row.items() if k is not None}
            try:
                row_id = int(str(row["ID"]).strip())
            except Exception:
                continue
            rows[row_id] = PromptRow(
                id=row_id,
                pair_code=str(row["Pair_Code"]).strip(),
                disease_pair=str(row["Disease_Pair"]).strip(),
                severity=str(row["Severity"]).strip(),
                climate=str(row["Climate"]).strip(),
                folder=str(row["Folder"]).strip() or str(row["Pair_Code"]).strip(),
            )
        return rows


def iter_images(input_dir: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        for p in input_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                yield p
        return
    for p in input_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            yield p


def parse_id_from_filename(path: Path, id_regex: re.Pattern[str]) -> Optional[int]:
    match = id_regex.search(path.stem)
    if not match:
        return None
    try:
        return int(match.group("id"))
    except Exception:
        return None


def center_crop_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width == height:
        return image
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def process_image(
    input_path: Path,
    output_path: Path,
    size: int,
    quality: int,
    crop_square: bool,
) -> None:
    with Image.open(input_path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        if crop_square:
            img = center_crop_square(img)
        img = img.resize((size, size), Image.Resampling.LANCZOS)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, format="JPEG", quality=quality, optimize=True)


def choose_output_path(
    input_path: Path,
    output_root: Path,
    prompt: Optional[PromptRow],
    file_id: Optional[int],
    slugify_folders: bool,
) -> Path:
    if prompt is None:
        rel = Path("unmapped")
    else:
        severity = _slugify(prompt.severity) if slugify_folders else prompt.severity
        climate = _slugify(prompt.climate) if slugify_folders else prompt.climate
        folder = _slugify(prompt.folder) if slugify_folders else prompt.folder
        rel = Path(folder) / severity / climate

    if file_id is not None:
        name = f"{file_id}.jpg"
    else:
        name = f"{input_path.stem}.jpg"
    return output_root / rel / name


def process_directory(
    input_dir: Path,
    output_dir: Path,
    prompt_csv: Optional[Path],
    recursive: bool,
    size: int,
    quality: int,
    crop_square: bool,
    slugify_folders: bool,
    id_regex: str,
    strict_id_mapping: bool,
    limit: Optional[int],
    manifest_path: Optional[Path],
) -> int:
    prompts: dict[int, PromptRow] = {}
    if prompt_csv is not None:
        prompts = load_prompts(prompt_csv)

    pattern = re.compile(id_regex)
    processed = 0
    failed = 0

    manifest_rows: list[dict[str, str]] = []

    for idx, img_path in enumerate(iter_images(input_dir, recursive=recursive)):
        if limit is not None and idx >= limit:
            break

        file_id = parse_id_from_filename(img_path, pattern)
        prompt = prompts.get(file_id) if file_id is not None else None
        if strict_id_mapping and prompt_csv is not None and prompt is None:
            failed += 1
            manifest_rows.append(
                {
                    "input": str(img_path),
                    "output": "",
                    "id": "" if file_id is None else str(file_id),
                    "status": "skipped_unmapped_id",
                    "error": "filename did not map to any ID in prompt.csv",
                }
            )
            continue

        out_path = choose_output_path(
            input_path=img_path,
            output_root=output_dir,
            prompt=prompt,
            file_id=file_id,
            slugify_folders=slugify_folders,
        )

        try:
            process_image(
                input_path=img_path,
                output_path=out_path,
                size=size,
                quality=quality,
                crop_square=crop_square,
            )
            processed += 1
            manifest_rows.append(
                {
                    "input": str(img_path),
                    "output": str(out_path),
                    "id": "" if file_id is None else str(file_id),
                    "status": "ok",
                    "error": "",
                }
            )
        except Exception as exc:  # pragma: no cover
            failed += 1
            manifest_rows.append(
                {
                    "input": str(img_path),
                    "output": str(out_path),
                    "id": "" if file_id is None else str(file_id),
                    "status": "failed",
                    "error": repr(exc),
                }
            )

    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["input", "output", "id", "status", "error"]
            )
            writer.writeheader()
            writer.writerows(manifest_rows)

    print(f"Processed: {processed}", file=sys.stderr)
    print(f"Failed/skipped: {failed}", file=sys.stderr)
    return 0 if failed == 0 else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Post-process generated images to match MLTSDC/MLTSDC-style training format "
            "(RGB, 256x256, JPEG quality 90) and optionally organize by prompt.csv."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory containing raw/generated images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory to write processed images into.",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=None,
        help="Optional path to prompt.csv for organizing output folders by ID/Severity/Climate.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for images under --input.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="Final image size (square). Default: 256.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=90,
        help="JPEG quality. Default: 90.",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Do not center-crop to square before resizing.",
    )
    parser.add_argument(
        "--no-slugify",
        action="store_true",
        help="Keep folder names as-is (may include spaces).",
    )
    parser.add_argument(
        "--id-regex",
        default=r"^(?P<id>\d+)",
        help=(
            "Regex used to extract an ID from the filename stem. Must contain a named group 'id'. "
            "Default matches a leading integer: ^(?P<id>\\d+)"
        ),
    )
    parser.add_argument(
        "--strict-id-mapping",
        action="store_true",
        help="If --prompts is set, skip files whose ID cannot be mapped to prompt.csv.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N images (debugging).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path to write a CSV manifest of processed files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.input.exists():
        parser.error(f"--input does not exist: {args.input}")

    crop_square = not args.no_crop
    slugify_folders = not args.no_slugify

    return process_directory(
        input_dir=args.input,
        output_dir=args.output,
        prompt_csv=args.prompts,
        recursive=args.recursive,
        size=args.size,
        quality=args.quality,
        crop_square=crop_square,
        slugify_folders=slugify_folders,
        id_regex=args.id_regex,
        strict_id_mapping=args.strict_id_mapping,
        limit=args.limit,
        manifest_path=args.manifest,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
