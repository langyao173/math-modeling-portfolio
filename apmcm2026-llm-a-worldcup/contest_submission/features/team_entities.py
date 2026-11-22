"""Team aliases and explicit historical-entity policies.

Column-name localization and team-entity normalization are intentionally kept
separate. The default ``conservative`` policy only merges spelling/language
aliases and the uncontroversial Germany FR -> Germany continuation. The
``legacy_merge`` policy is available solely for the documented sensitivity
analysis; it additionally inherits selected predecessor records.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TeamResolution:
    original: str
    canonical: str
    policy: str
    rule: str


TEAM_ALIASES = {
    "IR Iran": "Iran",
    "Iran IR": "Iran",
    "Korea Republic": "South Korea",
    "Korea Rep.": "South Korea",
    "United States": "USA",
    "United States of America": "USA",
    "U.S.A.": "USA",
    "Türkiye": "Turkey",
    "Côte d'Ivoire": "Cote d'Ivoire",
    "Ivory Coast": "Cote d'Ivoire",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Trinidad & Tobago": "Trinidad and Tobago",
    "China PR": "China",
    "PR China": "China",
    "德国": "Germany",
    "法国": "France",
    "巴西": "Brazil",
    "阿根廷": "Argentina",
    "西班牙": "Spain",
    "荷兰": "Netherlands",
    "英格兰": "England",
    "葡萄牙": "Portugal",
    "意大利": "Italy",
    "乌拉圭": "Uruguay",
    "比利时": "Belgium",
    "克罗地亚": "Croatia",
    "墨西哥": "Mexico",
    "美国": "USA",
    "加拿大": "Canada",
    "日本": "Japan",
    "韩国": "South Korea",
    "伊朗": "Iran",
    "澳大利亚": "Australia",
    "瑞士": "Switzerland",
    "丹麦": "Denmark",
    "瑞典": "Sweden",
    "波兰": "Poland",
    "塞尔维亚": "Serbia",
    "摩洛哥": "Morocco",
    "喀麦隆": "Cameroon",
    "尼日利亚": "Nigeria",
    "南非": "South Africa",
    "沙特阿拉伯": "Saudi Arabia",
    "哥伦比亚": "Colombia",
    "智利": "Chile",
    "秘鲁": "Peru",
    "厄瓜多尔": "Ecuador",
    "巴拉圭": "Paraguay",
    "哥斯达黎加": "Costa Rica",
    "俄罗斯": "Russia",
    "乌克兰": "Ukraine",
    "捷克": "Czech Republic",
}


CONSERVATIVE_INHERITANCE = {
    "Germany FR": "Germany",
}

LEGACY_INHERITANCE = {
    **CONSERVATIVE_INHERITANCE,
    "Soviet Union": "Russia",
    "Czechoslovakia": "Czech Republic",
    "Yugoslavia": "Serbia",
    "Zaire": "Congo DR",
}


def resolve_team_name(value, policy: str = "conservative") -> TeamResolution:
    """Normalize one team name and record the rule that was applied."""
    original = "" if pd.isna(value) else str(value).strip()
    if not original:
        return TeamResolution(original=original, canonical="", policy=policy, rule="empty")
    if policy not in {"conservative", "legacy_merge"}:
        raise ValueError("球队历史实体策略必须为 conservative 或 legacy_merge。")
    if original in TEAM_ALIASES:
        return TeamResolution(
            original=original,
            canonical=TEAM_ALIASES[original],
            policy=policy,
            rule="language_or_spelling_alias",
        )
    inheritance = CONSERVATIVE_INHERITANCE if policy == "conservative" else LEGACY_INHERITANCE
    if original in inheritance:
        return TeamResolution(
            original=original,
            canonical=inheritance[original],
            policy=policy,
            rule="historical_inheritance",
        )
    return TeamResolution(original=original, canonical=original, policy=policy, rule="identity")


def clean_team_name(value, policy: str = "conservative") -> str:
    return resolve_team_name(value, policy=policy).canonical


__all__ = [
    "CONSERVATIVE_INHERITANCE",
    "LEGACY_INHERITANCE",
    "TEAM_ALIASES",
    "TeamResolution",
    "clean_team_name",
    "resolve_team_name",
]
