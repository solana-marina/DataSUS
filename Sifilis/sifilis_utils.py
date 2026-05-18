from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from IPython.display import Markdown, display

try:
    from pysus.preprocessing.decoders import decodifica_idade_SINAN
except Exception:  # pragma: no cover - notebook fallback
    decodifica_idade_SINAN = None


TIPOS_SIFILIS = {
    "SIFA": "Sífilis adquirida",
    "SIFG": "Sífilis em gestante",
    "SIFC": "Sífilis congênita",
}

CODIGOS_CIDADES = {
    291360: "Ilhéus",
    291480: "Itabuna",
}

ANOS_ESPERADOS = list(range(2014, 2025))

COLUNAS_INTERESSE = [
    "TP_NOT",
    "ID_AGRAVO",
    "DT_NOTIFIC",
    "SEM_NOT",
    "NU_ANO",
    "SG_UF_NOT",
    "ID_MUNICIP",
    "ID_REGIONA",
    "DT_DIAG",
    "SEM_DIAG",
    "NU_IDADE_N",
    "CS_SEXO",
    "CS_GESTANT",
    "CS_RACA",
    "CS_ESCOL_N",
    "SG_UF",
    "ID_MN_RESI",
    "ID_RG_RESI",
    "ID_PAIS",
    "DT_INVEST",
    "ID_OCUPA_N",
    "CLASSI_FIN",
    "CRITERIO",
    "DOENCA_TRA",
    "EVOLUCAO",
    "DT_OBITO",
    "DT_ENCERRA",
    "PRE_UFREL",
    "PRE_MUNIRE",
    "TPEVIDENCI",
    "TPTESTE1",
    "DSTITULO1",
    "DTTESTE1",
    "TPCONFIRMA",
    "TPESQUEMA",
    "DSMOTIVO",
    "TPMOTPARC",
    "TPESQPAR",
    "TRATPARC",
    "ANT_IDADE",
    "ANT_RACA",
    "ESCOLMAE",
    "ANT_PRE_NA",
    "UF_PRE_NAT",
    "MUN_PRE_NA",
    "ANTSIFIL_N",
    "LAB_PARTO",
    "LAB_TITU_2",
    "LAB_DT3",
    "LAB_CONF",
    "TRA_ESQUEM",
    "TRA_DT",
    "ANT_TRATAD",
    "ANT_UF_CRI",
    "ANT_MUNI_C",
    "LABC_SANGU",
    "LABC_TIT_1",
    "LABC_DT_1",
    "LABC_IGG",
    "LABC_DT",
    "LABC_LIQUO",
    "LABC_TIT_2",
    "LABC_DT_2",
    "LABC_TITUL",
    "LABC_EVIDE",
    "LABC_LIQ_1",
    "TRA_DIAG_T",
    "TRA_ESQU_1",
    "DS_ESQUEMA",
    "EVO_DIAG_N",
    "TRA_DIAG_C",
    "CLI_ICTERI",
    "CLI_RINITE",
    "CLI_ANEMIA",
    "CLI_ESPLEN",
    "HEPATO",
    "CLI_OSTEO",
    "LESOES",
    "CLI_OUTRO",
    "SIN_OUTR_E",
    "CLI_PSEUDO",
]

SEXO_MAP = {
    "M": "Masculino",
    "F": "Feminino",
    "I": "Ignorado",
}

RACA_MAP = {
    "1": "Branca",
    "2": "Preta",
    "3": "Amarela",
    "4": "Parda",
    "5": "Indígena",
    "9": "Ignorado",
}

ESCOLARIDADE_MAP = {
    "0": "Analfabeto",
    "1": "1ª a 4ª série incompleta",
    "2": "4ª série completa",
    "3": "5ª a 8ª série incompleta",
    "4": "Ensino fundamental completo",
    "5": "Ensino médio incompleto",
    "6": "Ensino médio completo",
    "7": "Superior incompleto",
    "8": "Superior completo",
    "9": "Ignorado",
    "10": "Não se aplica",
}

GESTANTE_MAP = {
    "1": "1º trimestre",
    "2": "2º trimestre",
    "3": "3º trimestre",
    "4": "Idade gestacional ignorada",
    "5": "Não",
    "6": "Não se aplica",
    "9": "Ignorado",
}

CLASSIFICACAO_MAP = {
    "1": "Confirmado",
    "2": "Descartado",
    "8": "Inconclusivo",
    "9": "Ignorado",
}

CRITERIO_MAP = {
    "1": "Laboratorial",
    "2": "Clínico-epidemiológico",
    "3": "Clínico",
    "9": "Ignorado",
}

EVOLUCAO_MAP = {
    "1": "Vivo",
    "2": "Óbito pelo agravo",
    "3": "Óbito por outra causa",
    "5": "Aborto ou natimorto",
    "9": "Ignorado",
}

SIM_NAO_MAP = {
    "1": "Sim",
    "2": "Não",
    "3": "Não realizado",
    "4": "Não se aplica",
    "9": "Ignorado",
}

POPULACAO_CIDADES = {
    "Ilhéus": {
        2014: 161268,
        2015: 163220,
        2016: 164653,
        2017: 165257,
        2018: 165048,
        2019: 165392,
        2020: 166221,
        2021: 166149,
        2022: 166187,
        2023: 166500,
        2024: 166800,
    },
    "Itabuna": {
        2014: 210557,
        2015: 212740,
        2016: 214476,
        2017: 215853,
        2018: 222127,
        2019: 222693,
        2020: 223750,
        2021: 224123,
        2022: 224561,
        2023: 225000,
        2024: 225400,
    },
}

TABELAS_CACHE = {
    "sifilis_bahia": "sifilis_bahia.csv",
    "sifilis_ilheus_itabuna": "sifilis_ilheus_itabuna.csv",
    "sifilis_resumo_arquivos": "sifilis_resumo_arquivos.csv",
    "sifilis_cidades_ano": "sifilis_cidades_ano.csv",
    "sifilis_cidades_ano_completo": "sifilis_cidades_ano_completo.csv",
    "sifilis_cidades_mes": "sifilis_cidades_mes.csv",
    "sifilis_bahia_ano": "sifilis_bahia_ano.csv",
    "sifilis_bahia_mes": "sifilis_bahia_mes.csv",
    "sifilis_qualidade_completude_campos": "sifilis_qualidade_completude_campos.csv",
    "sifilis_qualidade_datas": "sifilis_qualidade_datas.csv",
    "sifilis_qualidade_nao_informado": "sifilis_qualidade_nao_informado.csv",
    "sifilis_qualidade_checagem_arquivos": "sifilis_qualidade_checagem_arquivos.csv",
    "sifilis_comparacao_2024": "sifilis_comparacao_2024.csv",
    "sifilis_demo_resumo": "sifilis_demo_resumo.csv",
    "sifilis_demo_sexo": "sifilis_demo_sexo.csv",
    "sifilis_demo_faixa_etaria": "sifilis_demo_faixa_etaria.csv",
    "sifilis_demo_piramide": "sifilis_demo_piramide.csv",
    "sifilis_demo_raca_cor": "sifilis_demo_raca_cor.csv",
    "sifilis_demo_escolaridade": "sifilis_demo_escolaridade.csv",
    "sifilis_demo_gestacao": "sifilis_demo_gestacao.csv",
    "sifilis_demo_idade_histograma": "sifilis_demo_idade_histograma.csv",
    "sifilis_demo_idade_mae_histograma": "sifilis_demo_idade_mae_histograma.csv",
    "sifilis_demo_pre_natal_congenita": "sifilis_demo_pre_natal_congenita.csv",
    "sifilis_demo_idade_mediana_tipo": "sifilis_demo_idade_mediana_tipo.csv",
    "sifilis_representatividade_bahia": "sifilis_representatividade_bahia.csv",
    "sifilis_representatividade_bahia_tipo": "sifilis_representatividade_bahia_tipo.csv",
    "sifilis_taxa_100_mil_total": "sifilis_taxa_100_mil_total.csv",
    "sifilis_taxa_100_mil_por_tipo": "sifilis_taxa_100_mil_por_tipo.csv",
    "sifilis_municipios_residencia_atendidos": "sifilis_municipios_residencia_atendidos.csv",
    "sifilis_adquirida_classificacao_cidades": "sifilis_adquirida_classificacao_cidades.csv",
    "sifilis_sifa_criterio": "sifilis_sifa_criterio.csv",
    "sifilis_sifa_evolucao": "sifilis_sifa_evolucao.csv",
    "sifilis_sifa_doenca_trabalho": "sifilis_sifa_doenca_trabalho.csv",
    "sifilis_sifg_testes_tratamento": "sifilis_sifg_testes_tratamento.csv",
    "sifilis_sifg_parceria": "sifilis_sifg_parceria.csv",
    "sifilis_sifc_pre_natal_tratamento": "sifilis_sifc_pre_natal_tratamento.csv",
    "sifilis_sifc_sinais_clinicos": "sifilis_sifc_sinais_clinicos.csv",
    "sifilis_sifc_evolucao": "sifilis_sifc_evolucao.csv",
    "sifilis_campos_unidade_disponiveis": "sifilis_campos_unidade_disponiveis.csv",
    "sifilis_municipio_notificacao_residentes_cidades": "sifilis_municipio_notificacao_residentes_cidades.csv",
    "ibge_populacao_2014_2024": "ibge_populacao_2014_2024.csv",
    "ibge_municipios_bahia": "ibge_municipios_bahia.csv",
}

ARTEFATOS_CACHE = {
    **TABELAS_CACHE,
    "ibge_malha_bahia_municipios": "ibge_malha_bahia_municipios.geojson",
}


def encontrar_projeto() -> Path:
    candidatos = [Path.cwd(), *Path.cwd().parents]
    for candidato in candidatos:
        if (candidato / "Dados").exists() and (candidato / "Sifilis").exists():
            return candidato
        if (candidato / "DataSUS" / "Dados").exists():
            return candidato / "DataSUS"
    raise FileNotFoundError("Não foi possível localizar a pasta DataSUS com Dados.")


def configurar_caminhos() -> tuple[Path, Path, Path]:
    projeto = encontrar_projeto()
    pasta_sifilis = projeto / "Sifilis"
    pasta_dados = projeto / "Dados"
    pasta_saida = pasta_sifilis / "dados_processados"
    pasta_saida.mkdir(parents=True, exist_ok=True)
    if str(pasta_sifilis) not in sys.path:
        sys.path.insert(0, str(pasta_sifilis))
    return projeto, pasta_dados, pasta_saida


def codigo_limpo(valor: object) -> object:
    if pd.isna(valor):
        return pd.NA
    texto = str(valor).strip()
    if not texto or texto.lower() in {"nan", "none", "<na>"}:
        return pd.NA
    if texto.endswith(".0"):
        texto = texto[:-2]
    return texto.strip() or pd.NA


def limpar_codigo_serie(serie: pd.Series) -> pd.Series:
    return serie.map(codigo_limpo).astype("string")


def mapear_codigo(serie: pd.Series, mapa: dict[str, str], prefixo: str = "Código") -> pd.Series:
    codigos = limpar_codigo_serie(serie)
    return codigos.map(mapa).fillna(codigos.map(lambda valor: f"{prefixo} {valor}" if pd.notna(valor) else "Não informado"))


def parse_data_sinan(serie: pd.Series) -> pd.Series:
    texto = (
        serie.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    datas = pd.Series(pd.NaT, index=serie.index, dtype="datetime64[ns]")
    mascara_compacta = texto.str.fullmatch(r"\d{8}", na=False)
    datas.loc[mascara_compacta] = pd.to_datetime(
        texto.loc[mascara_compacta],
        format="%Y%m%d",
        errors="coerce",
    )
    datas.loc[~mascara_compacta] = pd.to_datetime(
        texto.loc[~mascara_compacta],
        format="mixed",
        dayfirst=True,
        errors="coerce",
    )
    return datas


def decodificar_idade(serie: pd.Series) -> pd.Series:
    valores = pd.to_numeric(serie, errors="coerce")
    resultado = pd.Series(np.nan, index=serie.index, dtype="float64")
    mascara = valores.notna()
    if decodifica_idade_SINAN is not None and mascara.any():
        try:
            resultado.loc[mascara] = decodifica_idade_SINAN(valores.loc[mascara])
            return resultado
        except Exception:
            pass

    texto = valores.astype("Int64").astype("string")
    unidade = texto.str[0]
    numero = pd.to_numeric(texto.str[1:], errors="coerce")
    resultado.loc[unidade.eq("4")] = numero.loc[unidade.eq("4")]
    resultado.loc[unidade.eq("3")] = numero.loc[unidade.eq("3")] / 12
    resultado.loc[unidade.eq("2")] = numero.loc[unidade.eq("2")] / 365
    resultado.loc[unidade.eq("1")] = numero.loc[unidade.eq("1")] / 8760
    return resultado


def faixa_etaria(serie: pd.Series) -> pd.Series:
    bins = [-np.inf, 0, 4, 9, 19, 29, 39, 49, 59, np.inf]
    labels = ["<1", "1-4", "5-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60+"]
    return pd.cut(serie, bins=bins, labels=labels)


def arquivos_sifilis(pasta_dados: Path) -> list[dict[str, object]]:
    registros: list[dict[str, object]] = []
    for agravo in TIPOS_SIFILIS:
        for ano in ANOS_ESPERADOS:
            caminho = pasta_dados / f"{agravo}BR{str(ano)[-2:]}.csv"
            registros.append({"agravo": agravo, "ano": ano, "caminho": caminho, "existe": caminho.exists()})
    return registros


def carregar_arquivo(caminho: Path, agravo: str, ano: int) -> pd.DataFrame:
    cabecalho = pd.read_csv(caminho, nrows=0).columns.tolist()
    colunas = [coluna for coluna in COLUNAS_INTERESSE if coluna in cabecalho]
    df = pd.read_csv(caminho, usecols=colunas, low_memory=False)
    df["agravo_codigo"] = agravo
    df["tipo_sifilis"] = TIPOS_SIFILIS[agravo]
    df["arquivo"] = caminho.name
    df["ano_arquivo"] = ano
    return df


def mascara_residencia_bahia(df: pd.DataFrame) -> pd.Series:
    if "ID_MN_RESI" not in df.columns:
        return pd.Series(False, index=df.index)
    return limpar_codigo_serie(df["ID_MN_RESI"]).str.startswith("29", na=False)


def padronizar_base(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for coluna in COLUNAS_INTERESSE:
        if coluna not in df.columns:
            df[coluna] = pd.NA

    df["data_notificacao"] = parse_data_sinan(df["DT_NOTIFIC"])
    df["data_diagnostico"] = parse_data_sinan(df["DT_DIAG"])
    df["ano_notificacao"] = df["data_notificacao"].dt.year.fillna(pd.to_numeric(df["NU_ANO"], errors="coerce")).astype("Int64")
    df["mes_notificacao"] = df["data_notificacao"].dt.month.astype("Int64")
    df["periodo_notificacao"] = df["data_notificacao"].dt.to_period("M").astype("string")

    df["municipio_residencia"] = pd.to_numeric(df["ID_MN_RESI"], errors="coerce").astype("Int64")
    df["municipio_notificacao"] = pd.to_numeric(df["ID_MUNICIP"], errors="coerce").astype("Int64")
    df["cidade"] = df["municipio_residencia"].map(CODIGOS_CIDADES).astype("string")
    df["uf_residencia"] = df["municipio_residencia"].astype("string").str[:2]
    df["reside_bahia"] = df["uf_residencia"].eq("29")
    df["cidade_foco"] = df["cidade"].fillna("Outros municípios da Bahia")

    df["idade_anos"] = decodificar_idade(df["NU_IDADE_N"])
    df["faixa_etaria"] = faixa_etaria(df["idade_anos"]).astype("string")
    df["idade_mae"] = pd.to_numeric(df["ANT_IDADE"], errors="coerce")
    df["faixa_etaria_mae"] = faixa_etaria(df["idade_mae"]).astype("string")

    df["sexo"] = mapear_codigo(df["CS_SEXO"], SEXO_MAP)
    df["raca_cor"] = mapear_codigo(df["CS_RACA"], RACA_MAP)
    df["raca_cor_mae"] = mapear_codigo(df["ANT_RACA"], RACA_MAP)
    df["escolaridade"] = mapear_codigo(df["CS_ESCOL_N"], ESCOLARIDADE_MAP)
    df["escolaridade_mae"] = mapear_codigo(df["ESCOLMAE"], ESCOLARIDADE_MAP)
    df["gestante"] = mapear_codigo(df["CS_GESTANT"], GESTANTE_MAP)
    df["classificacao_final"] = mapear_codigo(df["CLASSI_FIN"], CLASSIFICACAO_MAP)
    df["criterio_confirmacao"] = mapear_codigo(df["CRITERIO"], CRITERIO_MAP)
    df["evolucao"] = mapear_codigo(df["EVOLUCAO"], EVOLUCAO_MAP)

    for coluna in [
        "DOENCA_TRA",
        "TPEVIDENCI",
        "TPTESTE1",
        "TPCONFIRMA",
        "TPESQUEMA",
        "DSMOTIVO",
        "TPMOTPARC",
        "TPESQPAR",
        "TRATPARC",
        "ANT_PRE_NA",
        "ANTSIFIL_N",
        "LAB_PARTO",
        "LAB_CONF",
        "TRA_ESQUEM",
        "ANT_TRATAD",
        "LABC_SANGU",
        "LABC_IGG",
        "LABC_LIQUO",
        "LABC_EVIDE",
        "LABC_LIQ_1",
        "TRA_DIAG_T",
        "TRA_DIAG_C",
        "CLI_ICTERI",
        "CLI_RINITE",
        "CLI_ANEMIA",
        "CLI_ESPLEN",
        "HEPATO",
        "CLI_OSTEO",
        "LESOES",
        "CLI_OUTRO",
        "CLI_PSEUDO",
    ]:
        df[f"{coluna}_label"] = mapear_codigo(df[coluna], SIM_NAO_MAP)

    return df


def agregar_periodos(df: pd.DataFrame, dimensoes: Iterable[str]) -> pd.DataFrame:
    dimensoes = list(dimensoes)
    return (
        df.groupby(dimensoes, dropna=False)
        .size()
        .reset_index(name="casos")
        .sort_values(dimensoes)
        .reset_index(drop=True)
    )


def completar_serie_anual(df: pd.DataFrame, cidades: bool = True) -> pd.DataFrame:
    tipos = list(TIPOS_SIFILIS.values())
    cidades_lista = list(CODIGOS_CIDADES.values()) if cidades else ["Bahia"]
    base = pd.MultiIndex.from_product(
        [ANOS_ESPERADOS, cidades_lista, tipos],
        names=["ano_notificacao", "cidade", "tipo_sifilis"],
    ).to_frame(index=False)
    return base.merge(df, on=["ano_notificacao", "cidade", "tipo_sifilis"], how="left").assign(
        casos=lambda dados: dados["casos"].fillna(0).astype(int)
    )


def caminho_tabela(nome: str, pasta_saida: Path | str | None = None) -> Path:
    _, _, saida_padrao = configurar_caminhos()
    pasta = Path(pasta_saida) if pasta_saida is not None else saida_padrao
    try:
        arquivo = TABELAS_CACHE[nome]
    except KeyError as exc:
        raise KeyError(f"Tabela processada desconhecida: {nome}") from exc
    return pasta / arquivo


def caminho_artefato(nome: str, pasta_saida: Path | str | None = None) -> Path:
    _, _, saida_padrao = configurar_caminhos()
    pasta = Path(pasta_saida) if pasta_saida is not None else saida_padrao
    try:
        arquivo = ARTEFATOS_CACHE[nome]
    except KeyError as exc:
        raise KeyError(f"Artefato processado desconhecido: {nome}") from exc
    return pasta / arquivo


def verificar_cache_processado(
    pasta_saida: Path | str | None = None,
    nomes: Iterable[str] | None = None,
) -> dict[str, object]:
    nomes_esperados = list(nomes or ARTEFATOS_CACHE)
    faltantes = [
        nome for nome in nomes_esperados
        if not caminho_artefato(nome, pasta_saida).exists()
    ]
    return {
        "ok": not faltantes,
        "faltantes": faltantes,
        "total_esperado": len(nomes_esperados),
        "total_faltante": len(faltantes),
    }


def exigir_cache_processado(
    pasta_saida: Path | str | None = None,
    nomes: Iterable[str] | None = None,
) -> None:
    status = verificar_cache_processado(pasta_saida, nomes)
    if status["ok"]:
        return
    faltantes = ", ".join(status["faltantes"])
    raise FileNotFoundError(
        "Cache processado incompleto. Execute o notebook 01 primeiro. "
        f"Tabelas/artefatos faltantes: {faltantes}"
    )


def ler_tabela(
    nome: str,
    colunas: list[str] | None = None,
    pasta_saida: Path | str | None = None,
    **kwargs,
) -> pd.DataFrame:
    caminho = caminho_tabela(nome, pasta_saida)
    if not caminho.exists():
        raise FileNotFoundError(
            f"Tabela processada ausente: {caminho}. Execute o notebook 01 primeiro."
        )
    argumentos = {"low_memory": False}
    argumentos.update(kwargs)
    if colunas is not None:
        argumentos["usecols"] = colunas
    return pd.read_csv(caminho, **argumentos)


def carregar_base_cidades(
    pasta_saida: Path | str | None = None,
    colunas: list[str] | None = None,
) -> pd.DataFrame:
    return ler_tabela("sifilis_ilheus_itabuna", colunas=colunas, pasta_saida=pasta_saida)


def carregar_agregados_temporais(pasta_saida: Path | str | None = None) -> dict[str, pd.DataFrame]:
    nomes = [
        "sifilis_cidades_ano",
        "sifilis_cidades_ano_completo",
        "sifilis_cidades_mes",
        "sifilis_bahia_ano",
        "sifilis_bahia_mes",
        "sifilis_representatividade_bahia",
        "sifilis_representatividade_bahia_tipo",
        "sifilis_taxa_100_mil_total",
        "sifilis_taxa_100_mil_por_tipo",
    ]
    return {nome: ler_tabela(nome, pasta_saida=pasta_saida) for nome in nomes}


def carregar_agregados_qualidade(pasta_saida: Path | str | None = None) -> dict[str, pd.DataFrame]:
    nomes = [
        "sifilis_resumo_arquivos",
        "sifilis_qualidade_completude_campos",
        "sifilis_qualidade_datas",
        "sifilis_qualidade_nao_informado",
        "sifilis_comparacao_2024",
        "sifilis_qualidade_checagem_arquivos",
    ]
    return {nome: ler_tabela(nome, pasta_saida=pasta_saida) for nome in nomes}


def carregar_agregados_demograficos(pasta_saida: Path | str | None = None) -> dict[str, pd.DataFrame]:
    nomes = [
        "sifilis_demo_resumo",
        "sifilis_demo_sexo",
        "sifilis_demo_faixa_etaria",
        "sifilis_demo_piramide",
        "sifilis_demo_raca_cor",
        "sifilis_demo_escolaridade",
        "sifilis_demo_gestacao",
        "sifilis_demo_idade_histograma",
        "sifilis_demo_idade_mae_histograma",
        "sifilis_demo_pre_natal_congenita",
        "sifilis_demo_idade_mediana_tipo",
    ]
    return {nome: ler_tabela(nome, pasta_saida=pasta_saida) for nome in nomes}


def carregar_agregados_especificos(pasta_saida: Path | str | None = None) -> dict[str, pd.DataFrame]:
    nomes = [
        "sifilis_adquirida_classificacao_cidades",
        "sifilis_sifa_criterio",
        "sifilis_sifa_evolucao",
        "sifilis_sifa_doenca_trabalho",
        "sifilis_sifg_testes_tratamento",
        "sifilis_sifg_parceria",
        "sifilis_sifc_pre_natal_tratamento",
        "sifilis_sifc_sinais_clinicos",
        "sifilis_sifc_evolucao",
        "sifilis_campos_unidade_disponiveis",
        "sifilis_municipio_notificacao_residentes_cidades",
        "sifilis_municipios_residencia_atendidos",
    ]
    return {nome: ler_tabela(nome, pasta_saida=pasta_saida) for nome in nomes}


def _contar_por_cidade(df: pd.DataFrame, campo: str) -> pd.DataFrame:
    return (
        df.groupby(["cidade", campo], dropna=False)
        .size()
        .reset_index(name="registros")
        .sort_values(["cidade", "registros"], ascending=[True, False])
        .reset_index(drop=True)
    )


def _contar_por_tipo(df: pd.DataFrame, campo: str) -> pd.DataFrame:
    return (
        df.groupby(["tipo_sifilis", campo], dropna=False)
        .size()
        .reset_index(name="registros")
        .rename(columns={campo: "categoria"})
        .sort_values(["tipo_sifilis", "registros"], ascending=[True, False])
        .reset_index(drop=True)
    )


def _contar_campo_longo(df: pd.DataFrame, campos: list[str]) -> pd.DataFrame:
    registros = []
    for campo in campos:
        if campo not in df.columns:
            continue
        contagem = _contar_por_cidade(df, campo)
        contagem = contagem.rename(columns={campo: "categoria"})
        contagem.insert(0, "campo", campo)
        registros.append(contagem)
    if not registros:
        return pd.DataFrame(columns=["campo", "cidade", "categoria", "registros"])
    return pd.concat(registros, ignore_index=True)


def _gerar_agregados_qualidade(
    cidades: pd.DataFrame,
    resumo: pd.DataFrame,
    cidades_ano_completo: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    campos = [
        "data_notificacao", "data_diagnostico", "idade_anos", "sexo", "raca_cor",
        "escolaridade", "gestante", "classificacao_final", "criterio_confirmacao",
        "evolucao", "idade_mae", "ANT_PRE_NA_label", "ANT_TRATAD_label",
        "TRATPARC_label",
    ]
    campos = [campo for campo in campos if campo in cidades.columns]
    registros = []
    for tipo, grupo in cidades.groupby("tipo_sifilis"):
        for campo in campos:
            valores = grupo[campo]
            ausentes = valores.isna() | valores.astype("string").str.strip().isin(["", "Não informado"])
            registros.append({"tipo_sifilis": tipo, "campo": campo, "ausencia_pct": ausentes.mean() * 100})
    completude = pd.DataFrame(registros)

    qualidade_datas = (
        cidades.assign(
            sem_data_notificacao=cidades["data_notificacao"].isna(),
            sem_periodo=cidades["periodo_notificacao"].isna(),
            ano_diferente_arquivo=(
                pd.to_numeric(cidades["ano_notificacao"], errors="coerce")
                != pd.to_numeric(cidades["ano_arquivo"], errors="coerce")
            ),
        )
        .groupby(["tipo_sifilis", "cidade"], as_index=False)
        .agg(
            registros=("arquivo", "count"),
            sem_data_notificacao=("sem_data_notificacao", "sum"),
            sem_periodo=("sem_periodo", "sum"),
            ano_diferente_arquivo=("ano_diferente_arquivo", "sum"),
        )
    )

    campos_categoria = ["sexo", "raca_cor", "escolaridade", "gestante", "classificacao_final", "evolucao"]
    registros = []
    for tipo, grupo in cidades.groupby("tipo_sifilis"):
        for campo in campos_categoria:
            if campo in grupo.columns:
                nao_info = grupo[campo].astype("string").str.contains(
                    "Não informado|Ignorado|Código <NA>", regex=True, na=False
                ).mean() * 100
                registros.append({"tipo_sifilis": tipo, "campo": campo, "nao_informado_pct": nao_info})
    nao_informado = pd.DataFrame(registros)

    comparacao_2024 = cidades_ano_completo.pivot_table(
        index=["cidade", "tipo_sifilis"],
        columns="ano_notificacao",
        values="casos",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    comparacao_2024["media_2021_2023"] = comparacao_2024[[2021, 2022, 2023]].mean(axis=1)
    comparacao_2024["razao_2024_media_2021_2023"] = np.where(
        comparacao_2024["media_2021_2023"] > 0,
        comparacao_2024[2024] / comparacao_2024["media_2021_2023"],
        np.nan,
    )

    checagem = (
        resumo.assign(arquivo_esperado=True)
        .groupby(["tipo_sifilis"], as_index=False)
        .agg(
            arquivos=("arquivo", "count"),
            primeiro_ano=("ano_arquivo", "min"),
            ultimo_ano=("ano_arquivo", "max"),
            linhas_bahia=("linhas_bahia_residencia", "sum"),
            linhas_cidades=("linhas_ilheus_itabuna", "sum"),
        )
    )

    return {
        "sifilis_qualidade_completude_campos": completude,
        "sifilis_qualidade_datas": qualidade_datas,
        "sifilis_qualidade_nao_informado": nao_informado,
        "sifilis_comparacao_2024": comparacao_2024,
        "sifilis_qualidade_checagem_arquivos": checagem,
    }


def _baixar_json(url: str) -> object:
    import requests

    resposta = requests.get(url, timeout=60)
    resposta.raise_for_status()
    return resposta.json()


def garantir_cache_ibge(pasta_saida: Path, force: bool = False) -> None:
    populacao_path = caminho_tabela("ibge_populacao_2014_2024", pasta_saida)
    municipios_path = caminho_tabela("ibge_municipios_bahia", pasta_saida)
    malha_path = caminho_artefato("ibge_malha_bahia_municipios", pasta_saida)

    if force or not populacao_path.exists():
        anos_estimativa = ",".join(str(ano) for ano in [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2024])
        localidades = [
            ("Bahia", "29", "n3", "29"),
            ("Ilhéus", "291360", "n6", "2913606"),
            ("Itabuna", "291480", "n6", "2914802"),
        ]
        registros = []
        for localidade, codigo_sinan, nivel, codigo_ibge in localidades:
            url = (
                "https://apisidra.ibge.gov.br/values/t/6579/"
                f"{nivel}/{codigo_ibge}/v/9324/p/{anos_estimativa}?formato=json"
            )
            for item in _baixar_json(url)[1:]:
                registros.append({
                    "localidade": localidade,
                    "codigo_sinan": codigo_sinan,
                    "ano": int(item["D3C"]),
                    "populacao": int(item["V"]),
                    "fonte": "IBGE/SIDRA tabela 6579 - População residente estimada",
                    "url": url,
                })

            url_2022 = (
                "https://apisidra.ibge.gov.br/values/t/4714/"
                f"{nivel}/{codigo_ibge}/v/93/p/2022?formato=json"
            )
            pop_2022 = int(_baixar_json(url_2022)[1]["V"])
            registros.append({
                "localidade": localidade,
                "codigo_sinan": codigo_sinan,
                "ano": 2022,
                "populacao": pop_2022,
                "fonte": "IBGE/SIDRA tabela 4714 - População residente do Censo 2022",
                "url": url_2022,
            })
            registros.append({
                "localidade": localidade,
                "codigo_sinan": codigo_sinan,
                "ano": 2023,
                "populacao": pop_2022,
                "fonte": "Censo 2022 usado como referência para 2023; EstimaPop/SIDRA 6579 não divulga 2023",
                "url": url_2022,
            })

        (
            pd.DataFrame(registros)
            .sort_values(["localidade", "ano"])
            .reset_index(drop=True)
            .to_csv(populacao_path, index=False)
        )

    if force or not municipios_path.exists():
        url = "https://servicodados.ibge.gov.br/api/v1/localidades/estados/29/municipios"
        registros = []
        for item in _baixar_json(url):
            codigo7 = str(item["id"])
            registros.append({
                "codigo_ibge7": codigo7,
                "codigo_sinan6": codigo7[:6],
                "municipio_nome": item["nome"],
                "fonte": "IBGE API Localidades",
                "url": url,
            })
        pd.DataFrame(registros).to_csv(municipios_path, index=False)

    if force or not malha_path.exists():
        url = (
            "https://servicodados.ibge.gov.br/api/v3/malhas/estados/29"
            "?formato=application/vnd.geo+json&qualidade=minima&intrarregiao=municipio"
        )
        malha = _baixar_json(url)
        for feature in malha["features"]:
            codigo7 = str(feature["properties"].get("codarea"))
            feature["properties"]["codigo_ibge7"] = codigo7
            feature["properties"]["codigo_sinan6"] = codigo7[:6]
            feature["properties"]["fonte"] = "IBGE API Malhas"
        malha_path.write_text(json.dumps(malha), encoding="utf-8")


def _gerar_agregados_demograficos(cidades: pd.DataFrame) -> dict[str, pd.DataFrame]:
    faixa_ordem = ["<1", "1-4", "5-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60+"]
    resumo = (
        cidades.groupby(["cidade", "tipo_sifilis"], as_index=False)
        .agg(registros=("arquivo", "count"), idade_mediana=("idade_anos", "median"))
    )
    sexo = _contar_por_tipo(cidades, "sexo")
    faixa = (
        cidades.groupby(["faixa_etaria", "tipo_sifilis"], dropna=False)
        .size()
        .reset_index(name="registros")
    )
    faixa["faixa_etaria"] = pd.Categorical(faixa["faixa_etaria"], faixa_ordem, ordered=True)
    faixa = faixa.sort_values(["faixa_etaria", "tipo_sifilis"]).reset_index(drop=True)

    piramide_base = cidades[cidades["sexo"].isin(["Masculino", "Feminino"])].copy()
    piramide = (
        piramide_base.groupby(["faixa_etaria", "sexo"], dropna=False)
        .size()
        .reset_index(name="registros")
    )
    piramide["valor"] = np.where(piramide["sexo"].eq("Masculino"), -piramide["registros"], piramide["registros"])

    raca = _contar_por_tipo(cidades, "raca_cor")
    escolaridade = _contar_por_tipo(cidades, "escolaridade")
    gestante = _contar_campo_longo(
        cidades[cidades["tipo_sifilis"].isin(["Sífilis adquirida", "Sífilis em gestante"])],
        ["gestante"],
    ).drop(columns="campo")

    idade_hist = (
        cidades.assign(idade_bin=pd.cut(cidades["idade_anos"], bins=list(range(0, 105, 5)), right=False).astype("string"))
        .groupby(["tipo_sifilis", "cidade", "idade_bin"], dropna=False)
        .size()
        .reset_index(name="registros")
    )

    congenita = cidades[cidades["tipo_sifilis"] == "Sífilis congênita"].copy()
    idade_mae_hist = (
        congenita.assign(idade_mae_bin=pd.cut(congenita["idade_mae"], bins=list(range(10, 55, 5)), right=False).astype("string"))
        .groupby(["cidade", "idade_mae_bin"], dropna=False)
        .size()
        .reset_index(name="registros")
    )
    pre_natal = _contar_campo_longo(congenita, ["ANT_PRE_NA_label"]).drop(columns="campo")
    idade_mediana = (
        cidades.groupby("tipo_sifilis", as_index=False)["idade_anos"]
        .median()
        .rename(columns={"idade_anos": "idade_mediana"})
        .sort_values("idade_mediana")
    )

    return {
        "sifilis_demo_resumo": resumo,
        "sifilis_demo_sexo": sexo,
        "sifilis_demo_faixa_etaria": faixa,
        "sifilis_demo_piramide": piramide,
        "sifilis_demo_raca_cor": raca,
        "sifilis_demo_escolaridade": escolaridade,
        "sifilis_demo_gestacao": gestante,
        "sifilis_demo_idade_histograma": idade_hist,
        "sifilis_demo_idade_mae_histograma": idade_mae_hist,
        "sifilis_demo_pre_natal_congenita": pre_natal,
        "sifilis_demo_idade_mediana_tipo": idade_mediana,
    }


def _gerar_agregados_comparacao(
    bahia: pd.DataFrame,
    cidades_ano: pd.DataFrame,
    bahia_ano: pd.DataFrame,
    pasta_saida: Path,
) -> dict[str, pd.DataFrame]:
    populacao = ler_tabela("ibge_populacao_2014_2024", pasta_saida=pasta_saida)

    cidades_total_tipo = (
        cidades_ano.groupby(["ano_notificacao", "tipo_sifilis"], as_index=False)["casos"].sum()
        .rename(columns={"casos": "casos_ilheus_itabuna"})
    )
    bahia_total_tipo = bahia_ano.rename(columns={"casos": "casos_bahia"})[
        ["ano_notificacao", "tipo_sifilis", "casos_bahia"]
    ]
    representatividade_tipo = cidades_total_tipo.merge(
        bahia_total_tipo, on=["ano_notificacao", "tipo_sifilis"], how="left"
    )
    representatividade_tipo["participacao_pct"] = (
        representatividade_tipo["casos_ilheus_itabuna"] / representatividade_tipo["casos_bahia"] * 100
    )

    cidades_total = (
        cidades_ano.groupby("ano_notificacao", as_index=False)["casos"].sum()
        .rename(columns={"casos": "casos_ilheus_itabuna"})
    )
    bahia_total = (
        bahia_ano.groupby("ano_notificacao", as_index=False)["casos"].sum()
        .rename(columns={"casos": "casos_bahia"})
    )
    representatividade = cidades_total.merge(bahia_total, on="ano_notificacao", how="left")
    representatividade["restante_bahia"] = representatividade["casos_bahia"] - representatividade["casos_ilheus_itabuna"]
    representatividade["participacao_pct"] = (
        representatividade["casos_ilheus_itabuna"] / representatividade["casos_bahia"] * 100
    )

    taxa_cidades = (
        cidades_ano.groupby(["ano_notificacao", "cidade"], as_index=False)["casos"].sum()
        .rename(columns={"cidade": "localidade"})
    )
    taxa_bahia = bahia_total.rename(columns={"casos_bahia": "casos"}).assign(localidade="Bahia")[
        ["ano_notificacao", "localidade", "casos"]
    ]
    taxa_total = pd.concat([taxa_cidades, taxa_bahia], ignore_index=True)
    taxa_total = taxa_total.merge(
        populacao.rename(columns={"ano": "ano_notificacao"}),
        on=["localidade", "ano_notificacao"],
        how="left",
    )
    taxa_total["taxa_100_mil"] = taxa_total["casos"] / taxa_total["populacao"] * 100000

    taxa_tipo = pd.concat(
        [
            cidades_ano.rename(columns={"cidade": "localidade"}),
            bahia_ano.rename(columns={"cidade": "localidade"}),
        ],
        ignore_index=True,
    )
    taxa_tipo = taxa_tipo.merge(
        populacao.rename(columns={"ano": "ano_notificacao"}),
        on=["localidade", "ano_notificacao"],
        how="left",
    )
    taxa_tipo["taxa_100_mil"] = taxa_tipo["casos"] / taxa_tipo["populacao"] * 100000

    municipios = ler_tabela("ibge_municipios_bahia", pasta_saida=pasta_saida, dtype={"codigo_sinan6": "string", "codigo_ibge7": "string"})
    atendimento = bahia.copy()
    atendimento["cidade_atendimento"] = pd.to_numeric(atendimento["municipio_notificacao"], errors="coerce").map(CODIGOS_CIDADES)
    atendimento = atendimento[atendimento["cidade_atendimento"].notna()].copy()
    atendimento["codigo_residencia_sinan6"] = (
        pd.to_numeric(atendimento["municipio_residencia"], errors="coerce")
        .astype("Int64")
        .astype("string")
    )
    origens = (
        atendimento.groupby(["cidade_atendimento", "codigo_residencia_sinan6"], dropna=False)
        .size()
        .reset_index(name="casos")
        .merge(municipios[["codigo_sinan6", "municipio_nome"]], left_on="codigo_residencia_sinan6", right_on="codigo_sinan6", how="left")
        .drop(columns=["codigo_sinan6"])
        .sort_values(["cidade_atendimento", "casos"], ascending=[True, False])
        .reset_index(drop=True)
    )

    return {
        "sifilis_representatividade_bahia": representatividade,
        "sifilis_representatividade_bahia_tipo": representatividade_tipo,
        "sifilis_taxa_100_mil_total": taxa_total,
        "sifilis_taxa_100_mil_por_tipo": taxa_tipo,
        "sifilis_municipios_residencia_atendidos": origens,
    }


def _gerar_agregados_especificos(
    cidades: pd.DataFrame,
    pasta_dados: Path,
    pasta_saida: Path,
) -> dict[str, pd.DataFrame]:
    sifa = cidades[cidades["tipo_sifilis"] == "Sífilis adquirida"].copy()
    sifg = cidades[cidades["tipo_sifilis"] == "Sífilis em gestante"].copy()
    sifc = cidades[cidades["tipo_sifilis"] == "Sífilis congênita"].copy()

    class_sifa = _contar_por_cidade(sifa, "classificacao_final")
    class_sifa["proporcao_pct"] = class_sifa.groupby("cidade")["registros"].transform(lambda s: s / s.sum() * 100)
    class_sifa = class_sifa.rename(columns={"registros": "casos"})

    campos_unidade = []
    for prefixo in TIPOS_SIFILIS:
        for caminho in sorted(pasta_dados.glob(f"{prefixo}BR*.csv")):
            colunas = pd.read_csv(caminho, nrows=0).columns.tolist()
            encontrados = [col for col in colunas if any(token in col.upper() for token in ["UNID", "CNES", "ESTAB"])]
            campos_unidade.append({
                "arquivo": caminho.name,
                "campos_unidade_encontrados": ", ".join(encontrados) if encontrados else "nenhum",
            })
    campos_unidade_df = pd.DataFrame(campos_unidade)

    municipios = ler_tabela("ibge_municipios_bahia", pasta_saida=pasta_saida, dtype={"codigo_sinan6": "string"})
    notificacao = cidades.copy()
    notificacao["codigo_notificacao_sinan6"] = (
        pd.to_numeric(notificacao["municipio_notificacao"], errors="coerce")
        .astype("Int64")
        .astype("string")
    )
    notificacao = (
        notificacao.groupby(["cidade", "codigo_notificacao_sinan6", "ano_notificacao"], dropna=False)
        .size()
        .reset_index(name="casos")
        .merge(municipios[["codigo_sinan6", "municipio_nome"]], left_on="codigo_notificacao_sinan6", right_on="codigo_sinan6", how="left")
        .drop(columns=["codigo_sinan6"])
    )

    sinais = [
        "CLI_ICTERI_label", "CLI_RINITE_label", "CLI_ANEMIA_label", "CLI_ESPLEN_label",
        "HEPATO_label", "CLI_OSTEO_label", "LESOES_label", "CLI_OUTRO_label", "CLI_PSEUDO_label",
    ]
    registros_sinais = []
    for cidade, grupo in sifc.groupby("cidade"):
        for campo in sinais:
            if campo in grupo.columns:
                registros_sinais.append({
                    "cidade": cidade,
                    "sinal": campo.replace("_label", ""),
                    "sim_pct": grupo[campo].eq("Sim").mean() * 100,
                })

    return {
        "sifilis_adquirida_classificacao_cidades": class_sifa,
        "sifilis_sifa_criterio": _contar_por_cidade(sifa, "criterio_confirmacao"),
        "sifilis_sifa_evolucao": _contar_por_cidade(sifa, "evolucao"),
        "sifilis_sifa_doenca_trabalho": _contar_por_cidade(sifa, "DOENCA_TRA_label"),
        "sifilis_sifg_testes_tratamento": _contar_campo_longo(sifg, ["TPEVIDENCI_label", "TPTESTE1_label", "TPCONFIRMA_label", "TPESQUEMA_label"]),
        "sifilis_sifg_parceria": _contar_campo_longo(sifg, ["TRATPARC_label", "TPESQPAR_label", "TPMOTPARC_label"]),
        "sifilis_sifc_pre_natal_tratamento": _contar_campo_longo(sifc, ["ANT_PRE_NA_label", "ANTSIFIL_N_label", "ANT_TRATAD_label", "LAB_PARTO_label"]),
        "sifilis_sifc_sinais_clinicos": pd.DataFrame(registros_sinais),
        "sifilis_sifc_evolucao": _contar_por_cidade(sifc, "evolucao"),
        "sifilis_campos_unidade_disponiveis": campos_unidade_df,
        "sifilis_municipio_notificacao_residentes_cidades": notificacao,
    }


def _salvar_tabelas(tabelas: dict[str, pd.DataFrame], pasta_saida: Path) -> None:
    for nome, tabela in tabelas.items():
        if nome not in TABELAS_CACHE:
            continue
        tabela.to_csv(caminho_tabela(nome, pasta_saida), index=False)


def preparar_dados_processados(pasta_dados: Path, pasta_saida: Path) -> dict[str, pd.DataFrame]:
    pasta_dados = Path(pasta_dados)
    pasta_saida = Path(pasta_saida)
    pasta_saida.mkdir(parents=True, exist_ok=True)
    garantir_cache_ibge(pasta_saida)

    inventario = pd.DataFrame(arquivos_sifilis(pasta_dados))
    if not inventario["existe"].all():
        faltantes = inventario.loc[~inventario["existe"], "caminho"].astype(str).tolist()
        raise FileNotFoundError(f"Arquivos ausentes: {faltantes}")

    partes_bahia = []
    resumo_arquivos = []
    for registro in inventario.itertuples(index=False):
        arquivo_df = carregar_arquivo(registro.caminho, registro.agravo, registro.ano)
        arquivo_bahia = arquivo_df.loc[mascara_residencia_bahia(arquivo_df)].copy()
        padrao_bahia = padronizar_base(arquivo_bahia)
        cidades_arquivo = padrao_bahia.loc[padrao_bahia["municipio_residencia"].isin(CODIGOS_CIDADES)]
        resumo_arquivos.append(
            {
                "arquivo": registro.caminho.name,
                "agravo_codigo": registro.agravo,
                "tipo_sifilis": TIPOS_SIFILIS[registro.agravo],
                "ano_arquivo": registro.ano,
                "linhas_nacionais": len(arquivo_df),
                "linhas_bahia_residencia": len(padrao_bahia),
                "linhas_ilheus_itabuna": len(cidades_arquivo),
            }
        )
        partes_bahia.append(padrao_bahia)

    bahia = pd.concat(partes_bahia, ignore_index=True)
    cidades = bahia.loc[bahia["municipio_residencia"].isin(CODIGOS_CIDADES)].copy()

    resumo = pd.DataFrame(resumo_arquivos)
    resumo["linhas_bahia_residencia"] = resumo["linhas_bahia_residencia"].astype(int)
    resumo["linhas_ilheus_itabuna"] = resumo["linhas_ilheus_itabuna"].astype(int)

    cidades_ano = agregar_periodos(cidades, ["ano_notificacao", "cidade", "tipo_sifilis"])
    cidades_mes = agregar_periodos(cidades.dropna(subset=["periodo_notificacao"]), ["periodo_notificacao", "cidade", "tipo_sifilis"])
    bahia_ano = agregar_periodos(bahia, ["ano_notificacao", "tipo_sifilis"])
    bahia_ano = bahia_ano.assign(cidade="Bahia")
    bahia_mes = agregar_periodos(bahia.dropna(subset=["periodo_notificacao"]), ["periodo_notificacao", "tipo_sifilis"])
    bahia_mes = bahia_mes.assign(cidade="Bahia")

    cidades_ano_completo = completar_serie_anual(cidades_ano, cidades=True)

    dados = {
        "sifilis_bahia": bahia,
        "sifilis_ilheus_itabuna": cidades,
        "sifilis_resumo_arquivos": resumo,
        "sifilis_cidades_ano": cidades_ano,
        "sifilis_cidades_ano_completo": cidades_ano_completo,
        "sifilis_cidades_mes": cidades_mes,
        "sifilis_bahia_ano": bahia_ano,
        "sifilis_bahia_mes": bahia_mes,
    }
    dados.update(_gerar_agregados_qualidade(cidades, resumo, cidades_ano_completo))
    dados.update(_gerar_agregados_demograficos(cidades))
    dados.update(_gerar_agregados_comparacao(bahia, cidades_ano, bahia_ano, pasta_saida))
    dados.update(_gerar_agregados_especificos(cidades, pasta_dados, pasta_saida))

    _salvar_tabelas(dados, pasta_saida)

    return dados


def processar_sifilis(
    force: bool = False,
    pasta_dados: Path | str | None = None,
    pasta_saida: Path | str | None = None,
) -> dict[str, pd.DataFrame] | dict[str, object]:
    _, dados_padrao, saida_padrao = configurar_caminhos()
    dados_dir = Path(pasta_dados) if pasta_dados is not None else dados_padrao
    saida_dir = Path(pasta_saida) if pasta_saida is not None else saida_padrao
    if not force:
        status = verificar_cache_processado(saida_dir)
        if status["ok"]:
            return {"cache_ok": True, "pasta_saida": saida_dir, "reprocessado": False}
    return preparar_dados_processados(dados_dir, saida_dir)


def carregar_processados(pasta_saida: Path) -> dict[str, pd.DataFrame]:
    nomes = [
        "sifilis_bahia",
        "sifilis_ilheus_itabuna",
        "sifilis_resumo_arquivos",
        "sifilis_cidades_ano",
        "sifilis_cidades_ano_completo",
        "sifilis_cidades_mes",
        "sifilis_bahia_ano",
        "sifilis_bahia_mes",
    ]
    return {nome: pd.read_csv(pasta_saida / f"{nome}.csv", low_memory=False) for nome in nomes}


def formatar_inteiro(valor: float | int) -> str:
    return f"{int(valor):,}".replace(",", ".")


def formatar_percentual(valor: float) -> str:
    return f"{valor:.1f}%".replace(".", ",")


def exibir_markdown(texto: str) -> None:
    display(Markdown(texto))


def salvar_figura(fig, pasta_saida: Path, nome: str) -> None:
    figuras = pasta_saida / "figuras"
    figuras.mkdir(parents=True, exist_ok=True)
    fig.savefig(figuras / nome, dpi=140, bbox_inches="tight")
