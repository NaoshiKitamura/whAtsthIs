"""
LLMに渡すプロンプトを組み立てるモジュール。

方針:
  - システムプロンプトで「計算材料科学に詳しいアシスタント」という役割と
    出力フォーマットの基本方針を固定する。
  - ユーザープロンプトには、抽出器が作った要約（summary）を渡し、
    生の全文は渡さない（巨大ファイル対策）。
  - カテゴリごとに「何を重点的に説明してほしいか」のヒントを追加する。
"""

from __future__ import annotations

from .detect import Category, Detection
from .extractors.base import ExtractionResult

SYSTEM_PROMPT = """\
あなたは計算材料科学・計算化学の研究者を支援するアシスタントです。
ユーザーは研究フォルダの中から1つのファイルを選び、その中身の要約（メタデータ抽出結果）を渡してきます。
あなたの仕事は、そのファイルについて次を過不足なく、簡潔かつ具体的な日本語で説明することです。

- これは何のファイルか（フォーマット・生成したソフトウェア等）
- このファイルは何をしている/何を表しているか
- どのような情報が含まれているか
- 研究の中でどのような役割・使われ方をするファイルか

出力は日本語のMarkdownで、見出しや箇条書きを使い、読みやすく簡潔にまとめてください。
渡された要約に無い情報を憶測で断定しないでください。不明な点は「不明」「情報からは判断できない」と正直に書いてください。
専門用語は初学者にも分かるよう軽く補足しつつ、冗長にならないようにしてください。
"""

_CATEGORY_HINTS: dict[Category, str] = {
    Category.PYTHON: (
        "これはPythonスクリプトです。次を含めてください: "
        "(1) スクリプト全体の目的, (2) 主な関数・クラスとその役割, "
        "(3) おおまかな実行フロー, (4) 使用している主要ライブラリとその用途, "
        "(5) 研究のどの工程（前処理/計算実行/後処理・解析/可視化など）に使われそうか。"
    ),
    Category.C: "Cのソースコードです。目的、主要関数、想定される用途（数値計算/ドライバ/ユーティリティ等）を説明してください。",
    Category.CPP: "C++のソースコードです。目的、主要クラス/関数、想定される用途を説明してください。",
    Category.FORTRAN: "Fortranのソースコードです。多くの場合、数値計算コード（第一原理計算・MD等）の一部です。module構成、主要なsubroutine/functionの役割、想定用途を説明してください。",
    Category.BASH: "シェルスクリプトです。ジョブ投入スクリプト・ワークフロー自動化スクリプトである可能性が高いです。何を実行しているか、どんなツールを呼び出しているかを説明してください。",
    Category.JSON: "設定ファイルまたはデータファイル(JSON)です。主要なキーの意味・用途を推測して説明してください。",
    Category.YAML: "設定ファイル(YAML)です。主要なキーの意味・用途（CI設定、計算パラメータ、環境定義など）を推測して説明してください。",
    Category.TOML: "設定ファイル(TOML)です。主要なキーの意味・用途（例: pyproject.toml ならPythonパッケージ設定）を説明してください。",
    Category.MARKDOWN: "Markdownドキュメントです。README/レポート/ノート等、どんな性質の文書かを見出し構造から推測して説明してください。",
    Category.VASP_OUTCAR: (
        "VASPのOUTCARファイルです。次を明確にしてください: "
        "(1) VASPの計算結果ログであること, (2) 静的計算/構造最適化/分子動力学(MD)のどれか, "
        "(3) 含まれる情報（エネルギー・力・応力・温度など）, "
        "(4) この計算がどんな研究目的（構造緩和、物性計算、MDシミュレーション等）に使われるものと考えられるか。"
    ),
    Category.VASP_VASPRUN: (
        "VASPのvasprun.xmlファイルです。OUTCARと同様の情報に加え、機械可読なXML形式であることの利点"
        "（例: pymatgen/ASE等での自動解析がしやすい）にも触れつつ説明してください。"
    ),
    Category.VASP_POSCAR: (
        "VASPの構造ファイル(POSCAR/CONTCAR)です。原子数、元素組成、セル情報、"
        "構造の特徴（結晶/表面スラブ/分子/欠陥構造の可能性など、推測できる範囲で）を説明してください。"
        "CONTCARの場合は「構造最適化やMDの最終・現在の構造を表す」ことにも触れてください。"
    ),
    Category.VASP_INCAR: "VASPの計算設定ファイル(INCAR)です。指定されているパラメータから、どんな種類の計算が意図されているか説明してください。",
    Category.VASP_KPOINTS: "VASPのk点サンプリング設定ファイル(KPOINTS)です。どのようなk点メッシュ/サンプリング方式かを説明してください。",
    Category.VASP_XDATCAR: "VASPのXDATCAR（原子座標の軌跡ファイル）です。MDシミュレーションや構造最適化過程の座標の時系列であることを説明してください。",
    Category.LAMMPS_LOG: (
        "LAMMPSのログファイルです。次を含めてください: "
        "(1) シミュレーションの設定（units, pair_style, アンサンブル）, "
        "(2) thermo出力から読み取れる実行内容（何ステップ実行したか、温度・エネルギーの推移）, "
        "(3) このログが研究の中でどう使われるか（物性評価、平衡化、生産run等）。"
    ),
    Category.LAMMPS_INPUT: "LAMMPSの入力スクリプトです。シミュレーションのセットアップ（系の定義、ポテンシャル、アンサンブル、実行内容）を説明してください。",
    Category.LAMMPS_DATA: "LAMMPSのdataファイル（原子配置・トポロジー定義ファイル）です。原子数・原子タイプ数・シミュレーションボックスサイズ等から系の規模・種類を説明してください。",
    Category.QE_INPUT: "Quantum ESPRESSOの入力ファイルです。&CONTROL, &SYSTEM等のnamelistから、どんな計算（scf, relax, md等）を意図しているか説明してください。",
    Category.STRUCTURE_CIF: "CIF形式の結晶構造ファイルです。組成・セル情報・対称性など、含まれる構造情報を説明してください。",
    Category.STRUCTURE_XYZ: "XYZ形式の構造ファイルです。原子数・元素組成・（複数フレームなら）軌跡である可能性を説明してください。",
    Category.STRUCTURE_EXTXYZ: "extended XYZ形式の構造ファイルです。座標に加えてエネルギー・力などの付加情報が含まれることが多く、機械学習ポテンシャルの訓練データ等に使われることがある点にも触れてください。",
}


def build_prompt(filename: str, detection: Detection, extraction: ExtractionResult) -> tuple[str, str]:
    hint = _CATEGORY_HINTS.get(detection.category, "")

    parts = [
        f"# 対象ファイル\n`{filename}`",
        f"# 検出されたカテゴリ\n{extraction.category_label}\n(判定理由: {detection.reason})",
    ]
    if hint:
        parts.append(f"# このカテゴリで特に説明してほしいこと\n{hint}")

    parts.append(f"# 抽出された要約情報\n```\n{extraction.summary}\n```")

    if extraction.raw_excerpt:
        parts.append(f"# 生テキストの抜粋（参考・全文ではない）\n```\n{extraction.raw_excerpt}\n```")

    if extraction.warnings:
        w = "\n".join(f"- {w}" for w in extraction.warnings)
        parts.append(f"# 抽出時の注意点\n{w}")

    user_prompt = "\n\n".join(parts)
    return SYSTEM_PROMPT, user_prompt
