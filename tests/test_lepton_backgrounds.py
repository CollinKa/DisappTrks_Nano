from pathlib import Path

from disapptrks.lepton_backgrounds import (
    estimate_lepton_background,
    write_lepton_background_json,
    write_lepton_background_latex,
)


def test_estimate_lepton_background_uses_an_product():
    counts = {
        "electron_background_control_NLayers4": 100.0,
        "electron_background_offline_NLayers4": 25.0,
        "electron_background_trigger_NLayers4": 20.0,
    }
    pair_counts = {
        "NLayers4": {
            "den_os": 50.0,
            "num_os": 5.0,
            "den_ss": 10.0,
            "num_ss": 1.0,
        }
    }

    estimates = estimate_lepton_background(
        flavor=r"$e$",
        layers=["NLayers4"],
        pair_counts=pair_counts,
        counts=counts,
        control_category="electron_background_control_{layer}",
        poffline_numerator_category="electron_background_offline_{layer}",
        poffline_denominator_category="electron_background_control_{layer}",
        pmiss_numerator_category="electron_background_trigger_{layer}",
        pmiss_denominator_category="electron_background_offline_{layer}",
    )

    estimate = estimates[0]
    assert estimate.p_veto.value == 0.1
    assert estimate.p_offline.value == 0.25
    assert estimate.p_miss.value == 0.8
    assert estimate.p_trigger.value == 0.8
    assert estimate.estimate.value == 2.0


def test_write_lepton_background_outputs(tmp_path: Path):
    estimates = estimate_lepton_background(
        flavor=r"$\mu$",
        layers=["NLayers4"],
        pair_counts={
            "NLayers4": {
                "den_os": 10.0,
                "num_os": 1.0,
                "den_ss": 0.0,
                "num_ss": 0.0,
            }
        },
        counts={
            "control_NLayers4": 10.0,
            "offline_NLayers4": 5.0,
            "trigger_NLayers4": 4.0,
        },
        control_category="control_{layer}",
        poffline_numerator_category="offline_{layer}",
        poffline_denominator_category="control_{layer}",
        pmiss_numerator_category="trigger_{layer}",
        pmiss_denominator_category="offline_{layer}",
    )

    json_path = tmp_path / "lepton.json"
    tex_path = tmp_path / "lepton.tex"
    write_lepton_background_json(estimates, json_path)
    write_lepton_background_latex(estimates, tex_path, run_period="2023D")

    assert '"p_miss"' in json_path.read_text()
    text = tex_path.read_text()
    assert "P_{\\mathrm{offline}}" in text
    assert "P_{\\mathrm{miss}}" in text
    assert "2023D" in text
