# Team review and independent work must be documented truthfully.
"""Aggregate-only reconstruction of the public validation analysis.

Read author workbook, retain sensory fields needed for pairing, and export
only counts/statistics. No participant identifiers or ancillary form metadata
are written to the output files.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ATTRIBUTES = ["Malt", "Sweetness", "Hops", "Bitterness", "Esters",
              "Fullness (body)", "Alcohol (warming)", "Overall appreciation"]
FIGURE_ORDER = ["Malt", "Hops", "Esters", "Sweetness", "Bitterness",
                "Alcohol (warming)", "Fullness (body)", "Overall appreciation"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", required=True)
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent))
    args = parser.parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    details = pd.read_excel(args.workbook, sheet_name="Validation Details")
    details = details[["Date", "Code", "BeerStyle", "SetName", "SampleName"]].copy()
    details["Date"] = details.Date.astype(str).str[:10]
    prepared = []
    for k in [1, 2, 3]:
        raw = pd.read_excel(args.workbook, sheet_name=f"Validation Results {k}")
        # Drop free text, browser, IP address, geolocation and form IDs before
        # doing any transformation. Name is used solely as a pairing key.
        raw = raw[["Time", "Name", "Beer"] + ATTRIBUTES].copy()
        raw["Date"] = raw.Time.astype(str).str[:10]
        used_keys = raw[["Date", "Beer"]].drop_duplicates().rename(columns={"Beer": "Code"})
        used_details = details.merge(used_keys, on=["Date", "Code"], how="inner").drop_duplicates()
        joined = raw.merge(used_details, left_on=["Date", "Beer"],
                           right_on=["Date", "Code"], how="left", validate="many_to_one")
        assert joined.SampleName.notna().all()
        data = joined[["BeerStyle", "SetName", "SampleName", "Name"] + ATTRIBUTES].drop_duplicates()
        assert not data.duplicated(["BeerStyle", "SetName", "SampleName", "Name"]).any()
        prepared.append(data)
    rows = []
    # The original figure processes the two Blond experiments together and
    # the non-alcoholic experiment separately; its alternating tie counter
    # resets between these two blocks, not within each attribute or set.
    for block in [pd.concat(prepared[:2], ignore_index=True), prepared[2]]:
        long = block.melt(id_vars=["BeerStyle", "SetName", "SampleName", "Name"],
                          value_vars=ATTRIBUTES, var_name="attribute", value_name="score")
        long["attribute"] = pd.Categorical(long.attribute, categories=FIGURE_ORDER, ordered=True)
        pair = long.pivot(index=["SetName", "Name", "attribute", "BeerStyle"],
                          columns="SampleName", values="score").reset_index()
        assert pair[["Control", "Spiked"]].notna().all().all()
        pair = pair.sort_values(["SetName", "Name", "attribute"], kind="stable").reset_index(drop=True)
        pair["difference"] = pair.Spiked - pair.Control
        pair["figure_difference"] = pair.difference
        tied = pair.loc[pair.difference == 0].sort_values("attribute", kind="stable").index
        pair.loc[tied, "figure_difference"] = np.where(np.arange(len(tied)) % 2 == 0, 1, -1)
        for (style, condition, attribute), group in pair.groupby(
                ["BeerStyle", "SetName", "attribute"], observed=True, sort=True):
            up = int((group.difference > 0).sum())
            down = int((group.difference < 0).sum())
            ties = int((group.difference == 0).sum())
            selected = int((group.figure_difference > 0).sum())
            total = len(group)
            rows.append(dict(beer_style=style, condition=condition, attribute=attribute,
                             n_pairs=total, raw_higher=up, raw_lower=down, raw_tied=ties,
                             figure_spiked=selected, figure_control=total-selected,
                             figure_spiked_fraction=selected/total,
                             figure_nominal_two_sided_binomial_p=binomtest(selected, total).pvalue,
                             non_tied_n=up+down,
                             excluding_ties_two_sided_sign_p=binomtest(up, up+down).pvalue if up+down else None))
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "spiking_validation_aggregate.csv", index=False)
    overall = frame.loc[frame.attribute == "Overall appreciation"]
    assert overall.n_pairs.tolist() == [20, 13, 13]
    assert overall.raw_higher.tolist() == [14, 11, 11]
    assert overall.raw_lower.tolist() == [5, 2, 2]
    assert overall.raw_tied.tolist() == [1, 0, 0]
    assert overall.figure_spiked.tolist() == [15, 11, 11]
    result = dict(
        sources={"workbook_sheets": ["Validation Results 1", "Validation Results 2", "Validation Results 3", "Validation Details"],
                 "analysis_script": "Main_Figure_Generator.Rmd, lines 731-787 and 934-964"},
        interpretation="Figure vote proportions are reconstructed from paired scores; the public plotting script alternates the sign of tied pairs after sorting sensory attributes.",
        test="Two-sided exact binomial test with p=0.5; exclusion of tied pairs is reported separately as a sensitivity analysis.",
        multiplicity="Nominal p-values as in the public figure script; no multiplicity correction is applied here.",
        privacy="Only aggregate counts and statistics exported; no participant keys or ancillary form metadata retained in outputs.",
        overall_appreciation=overall.to_dict(orient="records"))
    (output / "spiking_validation_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
