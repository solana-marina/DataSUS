from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from pandas.errors import EmptyDataError
from pysus import SINAN

AGRAVOS_ALVO = [
    "CHAG",
    "COQU",
    "DERM",
    "DIFT",
    "ESQU",
    "EXAN",
    "FMAC",
    "FTIF",
    "HANS",
    "HANT",
    "LEIV",
    "LEPT",
    "LTAN",
    "MALA",
    "MENI",
    "NTRA",
    "PEST",
    "PFAN",
    "RAIV",
    "SIFA",
    "SIFC",
    "SIFG",
    "SRC",
    "TETA",
    "TETN",
    "TRAC",
    "TUBE",
    "VARC",
]

ANOS_ALVO = list(range(2014, 2025))
CITY_CODES = {"Ilheus": 291360, "Itabuna": 291480}
DATA_DIR = Path("Dados")
PARQUET_DIR = Path.home() / "pysus"

DEFAULT_CITY_COLUMN = "ID_MN_RESI"
CITY_COLUMN_RULES = {
    "TRAC": "ID_MUNI_RE",
    "NTRA": "ID_MUNICIP",
}

AGRAVO_LABELS = {
    "CHAG": "Doenca de Chagas Aguda",
    "COQU": "Coqueluche",
    "DERM": "Dermatoses ocupacionais",
    "DIFT": "Difteria",
    "ESQU": "Esquistossomose",
    "EXAN": "Doencas exantematicas",
    "FMAC": "Febre Maculosa",
    "FTIF": "Febre Tifoide",
    "HANS": "Hanseniase",
    "HANT": "Hantavirose",
    "LEIV": "Leishmaniose Visceral",
    "LEPT": "Leptospirose",
    "LTAN": "Leishmaniose Tegumentar Americana",
    "MALA": "Malaria",
    "MENI": "Meningite",
    "NTRA": "Notificacao de Tracoma",
    "PEST": "Peste",
    "PFAN": "Paralisia Flacida Aguda",
    "RAIV": "Raiva",
    "SIFA": "Sifilis Adquirida",
    "SIFC": "Sifilis Congenita",
    "SIFG": "Sifilis em Gestante",
    "SRC": "Sindrome da Rubeola Congenita",
    "TETA": "Tetano Acidental",
    "TETN": "Tetano Neonatal",
    "TRAC": "Inquerito de Tracoma",
    "TUBE": "Tuberculose",
    "VARC": "Varicela",
}

STATUS_ORDER = {"present": 0, "reconvert": 1, "empty": 2, "missing": 3}


def build_expected_file_map(
    agravos_alvo: list[str] | None = None,
    anos_alvo: list[int] | None = None,
    catalog_timeout_seconds: float = 15.0,
) -> tuple[Any, dict[str, Any]]:
    agravos = list(agravos_alvo or AGRAVOS_ALVO)
    anos = list(anos_alvo or ANOS_ALVO)
    expected_total = len(agravos) * len(anos)

    expected_file_map = {
        f"{agravo}BR{str(ano)[-2:]}": None
        for agravo in agravos
        for ano in anos
    }

    sinan_db = None
    previous_timeout = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(catalog_timeout_seconds)
        sinan_db = SINAN().load()
        remote_files = sinan_db.get_files(dis_code=agravos, year=anos)
        for file_ref in remote_files:
            expected_file_map[Path(str(file_ref)).stem] = file_ref
    except Exception:
        sinan_db = None
    finally:
        socket.setdefaulttimeout(previous_timeout)

    if len(expected_file_map) != expected_total:
        raise ValueError(
            f"Esperava {expected_total} arquivos no escopo, "
            f"mas encontrei {len(expected_file_map)}."
        )
    return sinan_db, expected_file_map


def _inspect_csv(csv_path: Path) -> tuple[str, str]:
    if not csv_path.exists():
        return "missing", "csv not found"
    if csv_path.stat().st_size == 0:
        return "empty", "zero-byte csv"

    try:
        with csv_path.open("r", encoding="utf-8", errors="ignore") as handle:
            first_line = handle.readline()
            second_line = handle.readline()
    except OSError as exc:
        return "invalid", f"{type(exc).__name__}: {exc}"

    if not first_line:
        return "empty", "csv without content"

    try:
        header = pd.read_csv(csv_path, nrows=0)
    except EmptyDataError:
        return "empty", "csv without columns"
    except Exception as exc:
        return "invalid", f"{type(exc).__name__}: {exc}"

    if len(header.columns) == 0:
        return "empty", "csv without columns"
    if not second_line:
        return "empty", f"{len(header.columns)} columns and 0 data rows"
    return "present", f"{len(header.columns)} columns"


def _inspect_parquet(parquet_path: Path) -> tuple[str, str]:
    if not parquet_path.exists():
        return "missing", "parquet not found"
    if parquet_path.is_file() and parquet_path.stat().st_size == 0:
        return "empty", "zero-byte parquet"

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as exc:
        return "invalid", f"{type(exc).__name__}: {exc}"

    if df.shape[0] == 0 or df.shape[1] == 0:
        return "empty", f"{df.shape[1]} columns and {df.shape[0]} rows"
    return "present", f"{df.shape[1]} columns and {df.shape[0]} rows"


def _resolve_status(csv_state: str, parquet_state: str) -> str:
    if csv_state == "present":
        return "present"
    if parquet_state == "present":
        return "reconvert"
    if csv_state == "empty" or parquet_state == "empty":
        return "empty"
    return "missing"


def audit_downloads(
    expected_file_map: dict[str, Any],
    data_dir: Path | str = DATA_DIR,
    parquet_dir: Path | str = PARQUET_DIR,
) -> pd.DataFrame:
    data_dir = Path(data_dir)
    parquet_dir = Path(parquet_dir)
    records: list[dict[str, Any]] = []

    for stem in sorted(expected_file_map):
        code = stem[:-4]
        year = 2000 + int(stem[-2:])
        csv_path = data_dir / f"{stem}.csv"
        parquet_path = parquet_dir / f"{stem}.parquet"

        csv_state, csv_detail = _inspect_csv(csv_path)
        if csv_state == "present":
            parquet_state = "skipped"
            parquet_detail = "csv already valid"
            status = "present"
            needs_download = False
            needs_reconvert = False
        else:
            parquet_state, parquet_detail = _inspect_parquet(parquet_path)
            status = _resolve_status(csv_state, parquet_state)
            needs_download = status == "missing" or (status == "empty" and parquet_state != "empty")
            needs_reconvert = status == "reconvert"

        records.append(
            {
                "stem": stem,
                "agravo": code,
                "agravo_nome": AGRAVO_LABELS.get(code, code),
                "ano": year,
                "status": status,
                "needs_download": needs_download,
                "needs_reconvert": needs_reconvert,
                "csv_state": csv_state,
                "csv_detail": csv_detail,
                "parquet_state": parquet_state,
                "parquet_detail": parquet_detail,
                "csv_path": str(csv_path),
                "parquet_path": str(parquet_path),
                "remote_name": str(expected_file_map[stem] or f"{stem}.dbc"),
            }
        )

    audit_df = pd.DataFrame(records)
    if audit_df.empty:
        return audit_df

    return (
        audit_df.assign(status_rank=lambda df: df["status"].map(STATUS_ORDER))
        .sort_values(["status_rank", "agravo", "ano"])
        .drop(columns="status_rank")
        .reset_index(drop=True)
    )


def repair_downloads(
    sinan_db: Any,
    expected_file_map: dict[str, Any],
    audit_df: pd.DataFrame,
    data_dir: Path | str = DATA_DIR,
    parquet_dir: Path | str = PARQUET_DIR,
    download_timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    data_dir = Path(data_dir)
    parquet_dir = Path(parquet_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    parquet_dir.mkdir(parents=True, exist_ok=True)

    download_stems = audit_df.loc[audit_df["needs_download"], "stem"].tolist()
    reconvert_stems = audit_df.loc[audit_df["needs_reconvert"], "stem"].tolist()

    downloaded_paths: list[str] = []
    download_errors: list[dict[str, str]] = []
    for stem in download_stems:
        if sinan_db is None or expected_file_map.get(stem) is None:
            download_errors.append(
                {
                    "stem": stem,
                    "error": "remote catalog unavailable for download",
                }
            )
            continue
        previous_timeout = socket.getdefaulttimeout()
        try:
            socket.setdefaulttimeout(download_timeout_seconds)
            result = sinan_db.download([expected_file_map[stem]], local_dir=str(parquet_dir))
            downloaded_paths.extend(result)
        except Exception as exc:
            download_errors.append(
                {
                    "stem": stem,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            socket.setdefaulttimeout(previous_timeout)

    converted_stems: list[str] = []
    conversion_errors: list[dict[str, str]] = []
    for stem in sorted(set(download_stems + reconvert_stems)):
        parquet_path = parquet_dir / f"{stem}.parquet"
        csv_path = data_dir / f"{stem}.csv"
        if not parquet_path.exists():
            continue
        try:
            df = pd.read_parquet(parquet_path)
            df.to_csv(csv_path, index=False)
            converted_stems.append(stem)
        except Exception as exc:
            conversion_errors.append(
                {
                    "stem": stem,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    return {
        "downloaded_paths": downloaded_paths,
        "downloaded_stems": sorted({Path(path).stem for path in downloaded_paths}),
        "download_errors_df": pd.DataFrame(download_errors),
        "converted_stems": converted_stems,
        "conversion_errors_df": pd.DataFrame(conversion_errors),
    }


def choose_city_column(code: str, columns: list[str]) -> str | None:
    explicit_column = CITY_COLUMN_RULES.get(code)
    if explicit_column and explicit_column in columns:
        return explicit_column
    if DEFAULT_CITY_COLUMN in columns:
        return DEFAULT_CITY_COLUMN
    return None


def build_city_counts(
    audit_df: pd.DataFrame,
    city_codes: dict[str, int] | None = None,
    data_dir: Path | str = DATA_DIR,
    agravo_labels: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    city_codes = city_codes or CITY_CODES
    agravo_labels = agravo_labels or AGRAVO_LABELS
    data_dir = Path(data_dir)

    records: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    for row in audit_df.itertuples(index=False):
        if row.status == "present":
            csv_path = Path(row.csv_path)
            try:
                columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
            except Exception as exc:
                issues.append(
                    {
                        "stem": row.stem,
                        "agravo": row.agravo,
                        "ano": row.ano,
                        "issue": f"header_read_failed: {type(exc).__name__}: {exc}",
                    }
                )
                continue

            city_column = choose_city_column(row.agravo, columns)
            if city_column is None:
                issues.append(
                    {
                        "stem": row.stem,
                        "agravo": row.agravo,
                        "ano": row.ano,
                        "issue": "city_column_not_found",
                    }
                )
                continue

            city_series = pd.to_numeric(
                pd.read_csv(csv_path, usecols=[city_column], low_memory=False)[city_column],
                errors="coerce",
            )
            total_rows = int(len(city_series))
            for city_name, city_code in city_codes.items():
                records.append(
                    {
                        "agravo": row.agravo,
                        "agravo_nome": agravo_labels.get(row.agravo, row.agravo),
                        "ano": row.ano,
                        "cidade": city_name,
                        "casos": int((city_series == city_code).sum()),
                        "linhas_arquivo": total_rows,
                        "cidade_coluna": city_column,
                        "arquivo_status": row.status,
                    }
                )
            continue

        if row.status == "empty":
            city_column = CITY_COLUMN_RULES.get(row.agravo, DEFAULT_CITY_COLUMN)
            for city_name in city_codes:
                records.append(
                    {
                        "agravo": row.agravo,
                        "agravo_nome": agravo_labels.get(row.agravo, row.agravo),
                        "ano": row.ano,
                        "cidade": city_name,
                        "casos": 0,
                        "linhas_arquivo": 0,
                        "cidade_coluna": city_column,
                        "arquivo_status": row.status,
                    }
                )
            continue

        issues.append(
            {
                "stem": row.stem,
                "agravo": row.agravo,
                "ano": row.ano,
                "issue": f"skipped_status: {row.status}",
            }
        )

    city_counts_df = pd.DataFrame(records)
    if not city_counts_df.empty:
        city_counts_df = city_counts_df.sort_values(
            ["cidade", "ano", "agravo"]
        ).reset_index(drop=True)

    return city_counts_df, pd.DataFrame(issues)


def _build_top10(city_counts_df: pd.DataFrame, city_name: str | None = None) -> pd.DataFrame:
    if city_name is None:
        base_df = city_counts_df.groupby(
            ["agravo", "agravo_nome"], as_index=False
        )["casos"].sum()
    else:
        base_df = city_counts_df.loc[city_counts_df["cidade"] == city_name].groupby(
            ["agravo", "agravo_nome"], as_index=False
        )["casos"].sum()

    ranking_df = (
        base_df.sort_values(["casos", "agravo_nome"], ascending=[False, True])
        .head(10)
        .reset_index(drop=True)
    )
    ranking_df.insert(0, "posicao", range(1, len(ranking_df) + 1))
    ranking_df["agravo_exibicao"] = (
        ranking_df["agravo_nome"] + " (" + ranking_df["agravo"] + ")"
    )
    return ranking_df


def build_rankings(
    city_counts_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ranking_ilheus_df = _build_top10(city_counts_df, "Ilheus")
    ranking_itabuna_df = _build_top10(city_counts_df, "Itabuna")
    ranking_geral_df = _build_top10(city_counts_df, None)
    return ranking_ilheus_df, ranking_itabuna_df, ranking_geral_df


def plot_rankings(
    ranking_ilheus_df: pd.DataFrame,
    ranking_itabuna_df: pd.DataFrame,
    ranking_geral_df: pd.DataFrame,
) -> tuple[plt.Figure, Any]:
    fig, axes = plt.subplots(1, 3, figsize=(21, 8), constrained_layout=True)
    chart_config = [
        (axes[0], ranking_ilheus_df, "Top 10 - Ilheus", "#1f77b4"),
        (axes[1], ranking_itabuna_df, "Top 10 - Itabuna", "#ff7f0e"),
        (axes[2], ranking_geral_df, "Top 10 - Ilheus + Itabuna", "#2ca02c"),
    ]

    for ax, ranking_df, title, color in chart_config:
        if ranking_df.empty:
            ax.set_axis_off()
            ax.set_title(f"{title}\nsem dados")
            continue

        plot_df = ranking_df.sort_values("casos")
        ax.barh(plot_df["agravo_exibicao"], plot_df["casos"], color=color, alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel("Numero de notificacoes")
        ax.set_ylabel("")

        for y_pos, value in enumerate(plot_df["casos"]):
            ax.text(value, y_pos, f" {value}", va="center", fontsize=9)

    return fig, axes
