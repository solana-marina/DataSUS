from __future__ import annotations

import argparse
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
CITY_CODES = {"Ilhéus": 291360, "Itabuna": 291480}
DATA_DIR = Path("Dados")
PARQUET_DIR = Path.home() / "pysus"

DEFAULT_CITY_COLUMN = "ID_MN_RESI"
CITY_COLUMN_RULES = {
    "TRAC": "ID_MUNI_RE",
    "NTRA": "ID_MUNICIP",
}

DATE_COLUMN_CANDIDATES = [
    "DT_NOTIFIC",
    "DT_SIN_PRI",
    "DT_DIAG",
    "DT_INVEST",
]

AGRAVO_LABELS = {
    "CHAG": "Doença de Chagas Aguda",
    "COQU": "Coqueluche",
    "DERM": "Dermatoses ocupacionais",
    "DIFT": "Difteria",
    "ESQU": "Esquistossomose",
    "EXAN": "Doenças exantemáticas",
    "FMAC": "Febre Maculosa",
    "FTIF": "Febre Tifoide",
    "HANS": "Hanseníase",
    "HANT": "Hantavirose",
    "LEIV": "Leishmaniose Visceral",
    "LEPT": "Leptospirose",
    "LTAN": "Leishmaniose Tegumentar Americana",
    "MALA": "Malária",
    "MENI": "Meningite",
    "NTRA": "Notificação de Tracoma",
    "PEST": "Peste",
    "PFAN": "Paralisia Flácida Aguda",
    "RAIV": "Raiva",
    "SIFA": "Sífilis Adquirida",
    "SIFC": "Sífilis Congênita",
    "SIFG": "Sífilis em Gestante",
    "SRC": "Síndrome da Rubéola Congênita",
    "TETA": "Tétano Acidental",
    "TETN": "Tétano Neonatal",
    "TRAC": "Inquérito de Tracoma",
    "TUBE": "Tuberculose",
    "VARC": "Varicela",
}

STATUS_ORDER = {"present": 0, "reconvert": 1, "empty": 2, "missing": 3}


def build_expected_file_map(
    agravos_alvo: list[str] | None = None,
    anos_alvo: list[int] | None = None,
    catalog_timeout_seconds: float = 15.0,
    use_remote_catalog: bool = True,
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
    if use_remote_catalog:
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


def summarize_audit(audit_df: pd.DataFrame) -> pd.DataFrame:
    if audit_df.empty:
        return pd.DataFrame(columns=["status", "needs_download", "needs_reconvert", "arquivos"])

    return (
        audit_df.groupby(["status", "needs_download", "needs_reconvert"], as_index=False)
        .size()
        .rename(columns={"size": "arquivos"})
        .sort_values(["needs_download", "needs_reconvert", "status"], ascending=[False, False, True])
        .reset_index(drop=True)
    )


def run_data_update(
    agravos_alvo: list[str] | None = None,
    anos_alvo: list[int] | None = None,
    data_dir: Path | str = DATA_DIR,
    parquet_dir: Path | str = PARQUET_DIR,
    catalog_timeout_seconds: float = 15.0,
    download_timeout_seconds: float = 30.0,
    repair: bool = True,
) -> dict[str, Any]:
    sinan_db, expected_file_map = build_expected_file_map(
        agravos_alvo,
        anos_alvo,
        catalog_timeout_seconds=catalog_timeout_seconds,
        use_remote_catalog=repair,
    )
    initial_audit_df = audit_downloads(expected_file_map, data_dir, parquet_dir)

    repair_result = {
        "downloaded_paths": [],
        "downloaded_stems": [],
        "download_errors_df": pd.DataFrame(),
        "converted_stems": [],
        "conversion_errors_df": pd.DataFrame(),
    }
    if repair:
        repair_result = repair_downloads(
            sinan_db,
            expected_file_map,
            initial_audit_df,
            data_dir,
            parquet_dir,
            download_timeout_seconds=download_timeout_seconds,
        )

    final_audit_df = audit_downloads(expected_file_map, data_dir, parquet_dir)
    return {
        "sinan_db_available": sinan_db is not None,
        "expected_file_map": expected_file_map,
        "initial_audit_df": initial_audit_df,
        "initial_summary_df": summarize_audit(initial_audit_df),
        "repair_result": repair_result,
        "final_audit_df": final_audit_df,
        "final_summary_df": summarize_audit(final_audit_df),
    }


def _print_data_update_report(result: dict[str, Any]) -> None:
    expected_count = len(result["expected_file_map"])
    initial_pending = int(
        (
            result["initial_audit_df"]["needs_download"]
            | result["initial_audit_df"]["needs_reconvert"]
        ).sum()
    )
    final_pending = int(
        (
            result["final_audit_df"]["needs_download"]
            | result["final_audit_df"]["needs_reconvert"]
        ).sum()
    )
    repair_result = result["repair_result"]

    print(f"Arquivos esperados: {expected_count}")
    print(f"Catálogo remoto disponível: {result['sinan_db_available']}")
    print("\nResumo inicial:")
    print(result["initial_summary_df"].to_string(index=False))
    print(f"Pendências iniciais: {initial_pending}")

    print("\nCorreção incremental:")
    print(f"Baixados: {len(repair_result['downloaded_stems'])}")
    print(f"Convertidos: {len(repair_result['converted_stems'])}")
    if not repair_result["download_errors_df"].empty:
        print("\nFalhas de download:")
        print(repair_result["download_errors_df"].to_string(index=False))
    if not repair_result["conversion_errors_df"].empty:
        print("\nFalhas de conversão:")
        print(repair_result["conversion_errors_df"].to_string(index=False))

    print("\nResumo final:")
    print(result["final_summary_df"].to_string(index=False))
    print(f"Pendências finais: {final_pending}")

    pending_df = result["final_audit_df"].loc[
        result["final_audit_df"]["needs_download"]
        | result["final_audit_df"]["needs_reconvert"],
        ["stem", "status", "csv_state", "parquet_state"],
    ]
    if not pending_df.empty:
        print("\nArquivos ainda pendentes:")
        print(pending_df.to_string(index=False))


def choose_city_column(code: str, columns: list[str]) -> str | None:
    explicit_column = CITY_COLUMN_RULES.get(code)
    if explicit_column and explicit_column in columns:
        return explicit_column
    if DEFAULT_CITY_COLUMN in columns:
        return DEFAULT_CITY_COLUMN
    return None


def choose_date_column(columns: list[str]) -> str | None:
    for column in DATE_COLUMN_CANDIDATES:
        if column in columns:
            return column
    return None


def parse_sinan_dates(series: pd.Series) -> pd.Series:
    date_text = (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    parsed_dates = pd.to_datetime(pd.Series(pd.NaT, index=series.index))
    compact_mask = date_text.str.fullmatch(r"\d{8}", na=False)
    parsed_dates.loc[compact_mask] = pd.to_datetime(
        date_text.loc[compact_mask],
        format="%Y%m%d",
        errors="coerce",
    )
    parsed_dates.loc[~compact_mask] = pd.to_datetime(
        date_text.loc[~compact_mask],
        format="mixed",
        errors="coerce",
        dayfirst=True,
    )
    return parsed_dates


def build_monthly_city_counts(
    audit_df: pd.DataFrame,
    city_codes: dict[str, int] | None = None,
    data_dir: Path | str = DATA_DIR,
    agravo_labels: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    city_codes = city_codes or CITY_CODES
    agravo_labels = agravo_labels or AGRAVO_LABELS

    records: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    for row in audit_df.itertuples(index=False):
        if row.status == "empty":
            for city_name in city_codes:
                for period in pd.period_range(f"{row.ano}-01", f"{row.ano}-12", freq="M"):
                    records.append(
                        {
                            "agravo": row.agravo,
                            "agravo_nome": agravo_labels.get(row.agravo, row.agravo),
                            "ano": int(period.year),
                            "mes": int(period.month),
                            "periodo": str(period),
                            "cidade": city_name,
                            "casos": 0,
                            "cidade_coluna": CITY_COLUMN_RULES.get(row.agravo, DEFAULT_CITY_COLUMN),
                            "data_coluna": None,
                            "arquivo_status": row.status,
                        }
                    )
            continue

        if row.status != "present":
            issues.append(
                {
                    "stem": row.stem,
                    "agravo": row.agravo,
                    "ano": row.ano,
                    "issue": f"skipped_status: {row.status}",
                }
            )
            continue

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
        date_column = choose_date_column(columns)
        if city_column is None or date_column is None:
            issues.append(
                {
                    "stem": row.stem,
                    "agravo": row.agravo,
                    "ano": row.ano,
                    "issue": "city_or_date_column_not_found",
                }
            )
            continue

        try:
            file_df = pd.read_csv(csv_path, usecols=[city_column, date_column], low_memory=False)
        except Exception as exc:
            issues.append(
                {
                    "stem": row.stem,
                    "agravo": row.agravo,
                    "ano": row.ano,
                    "issue": f"read_failed: {type(exc).__name__}: {exc}",
                }
            )
            continue

        file_df[city_column] = pd.to_numeric(file_df[city_column], errors="coerce")
        file_df[date_column] = parse_sinan_dates(file_df[date_column])
        file_df = file_df.loc[file_df[date_column].notna()].copy()
        file_df["periodo"] = file_df[date_column].dt.to_period("M").astype(str)

        complete_periods = pd.period_range(f"{row.ano}-01", f"{row.ano}-12", freq="M").astype(str)
        for city_name, city_code in city_codes.items():
            city_series = (
                file_df.loc[file_df[city_column] == city_code]
                .groupby("periodo")
                .size()
                .reindex(complete_periods, fill_value=0)
            )
            for period, cases in city_series.items():
                period_obj = pd.Period(period, freq="M")
                records.append(
                    {
                        "agravo": row.agravo,
                        "agravo_nome": agravo_labels.get(row.agravo, row.agravo),
                        "ano": int(period_obj.year),
                        "mes": int(period_obj.month),
                        "periodo": str(period_obj),
                        "cidade": city_name,
                        "casos": int(cases),
                        "cidade_coluna": city_column,
                        "data_coluna": date_column,
                        "arquivo_status": row.status,
                    }
                )

    monthly_counts_df = pd.DataFrame(records)
    if not monthly_counts_df.empty:
        monthly_counts_df = monthly_counts_df.sort_values(
            ["cidade", "agravo", "ano", "mes"]
        ).reset_index(drop=True)

    return monthly_counts_df, pd.DataFrame(issues)


def build_candidate_inventory(monthly_counts_df: pd.DataFrame) -> pd.DataFrame:
    if monthly_counts_df.empty:
        return pd.DataFrame()

    expected_months = monthly_counts_df["periodo"].nunique()
    total_by_period = (
        monthly_counts_df.groupby(["agravo", "agravo_nome", "periodo"], as_index=False)["casos"]
        .sum()
        .sort_values(["agravo", "periodo"])
    )

    records: list[dict[str, Any]] = []
    for (agravo, agravo_nome), group in total_by_period.groupby(["agravo", "agravo_nome"]):
        values = group["casos"].astype(float)
        total_cases = int(values.sum())
        zero_months = int((values == 0).sum())
        active_months = int((values > 0).sum())
        first_half = group.loc[group["periodo"] <= "2018-12", "casos"].sum()
        second_half = group.loc[group["periodo"] >= "2019-01", "casos"].sum()
        month_means = (
            group.assign(mes=lambda df: pd.PeriodIndex(df["periodo"], freq="M").month)
            .groupby("mes")["casos"]
            .mean()
        )
        seasonality_ratio = 0.0
        if month_means.mean() > 0:
            seasonality_ratio = float(month_means.std(ddof=0) / month_means.mean())
        trend_delta = int(second_half - first_half)

        records.append(
            {
                "agravo": agravo,
                "agravo_nome": agravo_nome,
                "total_casos": total_cases,
                "media_mensal": round(float(values.mean()), 2),
                "max_mensal": int(values.max()),
                "meses_com_caso": active_months,
                "meses_zerados": zero_months,
                "pct_meses_zerados": round(zero_months / len(group) * 100, 1),
                "completude_mensal_pct": round(len(group) / expected_months * 100, 1),
                "casos_2014_2018": int(first_half),
                "casos_2019_2024": int(second_half),
                "variacao_abs_periodos": trend_delta,
                "indice_sazonalidade": round(seasonality_ratio, 2),
            }
        )

    inventory_df = pd.DataFrame(records)
    if inventory_df.empty:
        return inventory_df

    inventory_df["score_modelagem"] = (
        inventory_df["total_casos"].rank(pct=True) * 30
        + inventory_df["media_mensal"].rank(pct=True) * 15
        + (100 - inventory_df["pct_meses_zerados"]).rank(pct=True) * 20
        + inventory_df["meses_com_caso"].rank(pct=True) * 10
        + inventory_df["max_mensal"].rank(pct=True) * 10
        + inventory_df["variacao_abs_periodos"].abs().rank(pct=True) * 10
        + inventory_df["indice_sazonalidade"].rank(pct=True) * 5
    ).round(1)

    return inventory_df.sort_values(
        ["score_modelagem", "total_casos"],
        ascending=[False, False],
    ).reset_index(drop=True)


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
    ranking_ilheus_df = _build_top10(city_counts_df, "Ilhéus")
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
        (axes[0], ranking_ilheus_df, "Top 10 - Ilhéus", "#1f77b4"),
        (axes[1], ranking_itabuna_df, "Top 10 - Itabuna", "#ff7f0e"),
        (axes[2], ranking_geral_df, "Top 10 - Ilhéus + Itabuna", "#2ca02c"),
    ]

    for ax, ranking_df, title, color in chart_config:
        if ranking_df.empty:
            ax.set_axis_off()
            ax.set_title(f"{title}\nsem dados")
            continue

        plot_df = ranking_df.sort_values("casos")
        ax.barh(plot_df["agravo_exibicao"], plot_df["casos"], color=color, alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel("Número de notificações")
        ax.set_ylabel("")

        for y_pos, value in enumerate(plot_df["casos"]):
            ax.text(value, y_pos, f" {value}", va="center", fontsize=9)

    return fig, axes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audita, baixa e valida os arquivos SINAN usados no projeto."
    )
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Diretório dos CSVs.")
    parser.add_argument(
        "--parquet-dir",
        default=str(PARQUET_DIR),
        help="Diretório dos parquet baixados pelo PySUS.",
    )
    parser.add_argument(
        "--catalog-timeout",
        type=float,
        default=15.0,
        help="Timeout em segundos para carregar o catálogo remoto.",
    )
    parser.add_argument(
        "--download-timeout",
        type=float,
        default=30.0,
        help="Timeout em segundos para cada tentativa de download.",
    )
    parser.add_argument(
        "--skip-repair",
        action="store_true",
        help="Executa apenas a auditoria local, sem baixar ou converter arquivos.",
    )
    args = parser.parse_args()

    result = run_data_update(
        data_dir=args.data_dir,
        parquet_dir=args.parquet_dir,
        catalog_timeout_seconds=args.catalog_timeout,
        download_timeout_seconds=args.download_timeout,
        repair=not args.skip_repair,
    )
    _print_data_update_report(result)


if __name__ == "__main__":
    main()
