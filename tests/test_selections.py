import pytest

ak = pytest.importorskip("awkward")

from disapptrks.selections import (
    analysis_layer_mask,
    base_probe_track_mask,
    build_lepton_veto_tag_probe_pairs,
    build_muon_veto_tag_probe_pairs,
    fiducial_map_probe_track_mask,
    fake_track_base_mask,
    fake_track_layer_cut,
    fake_track_no_d0_mask,
    hadronic_tau_control_object_mask,
    layer_mask,
    lepton_veto_probe_track_mask,
    met_no_mu_minus_lepton,
    muon_veto_probe_track_cutflow_masks,
    muon_veto_probe_track_mask,
    probe_track_dedx_mask,
    search_track_cutflow_masks,
    search_track_mask,
    tau_veto_probe_track_cutflow_masks,
    tau_veto_probe_track_mask,
)


PROBE_TRACK_BASE_FIELDS = {
    "pt": 100.0, "eta": 0.8, "dxy": 0.01, "dz": 0.1,
    "inECALCrack": False, "inDTWheelGap": False,
    "inCSCTransition": False, "inTOBCrack": False,
    "isFiducialECALTrack": True, "hp_nValidPixelHits": 4,
    "hp_nValidHits": 4, "hp_nValidTrackerHits": 4,
    "missingInnerHits": 0, "missingMiddleHits": 0,
    "missingOuterHits": 3, "pfRelIso03_chg": 0.01, "dRMinJet": 1.0,
    "dRMinElectron": 1.0, "dRMinMuon": 1.0, "dRMinTauHad": 1.0,
    "caloEnergy": 1.0, "isHighPurityTrack": True,
    "hp_trackerLayersWithMeasurement": 4,
}


def test_analysis_layer_mask_requires_high_purity_for_every_layer_bin():
    tracks = ak.Array(
        [[
            {"hp_trackerLayersWithMeasurement": 4, "isHighPurityTrack": False},
            {"hp_trackerLayersWithMeasurement": 4, "isHighPurityTrack": True},
            {"hp_trackerLayersWithMeasurement": 5, "isHighPurityTrack": False},
            {"hp_trackerLayersWithMeasurement": 6, "isHighPurityTrack": False},
            {"hp_trackerLayersWithMeasurement": 5, "isHighPurityTrack": True},
            {"hp_trackerLayersWithMeasurement": 6, "isHighPurityTrack": True},
        ]]
    )

    assert ak.to_list(analysis_layer_mask(tracks, "combinedBins")) == [
        [False, True, False, False, True, True]
    ]
    assert ak.to_list(analysis_layer_mask(tracks, "NLayers4")) == [
        [False, True, False, False, False, False]
    ]
    assert ak.to_list(analysis_layer_mask(tracks, "NLayers5")) == [
        [False, False, False, False, True, False]
    ]
    assert ak.to_list(analysis_layer_mask(tracks, "NLayers6plus")) == [
        [False, False, False, False, False, True]
    ]


def test_high_purity_study_can_retain_non_high_purity_tracks():
    base = {
        "pt": 100.0, "eta": 0.8, "dxy": 0.1, "dz": 0.1,
        "inECALCrack": False, "inDTWheelGap": False,
        "inCSCTransition": False, "inTOBCrack": False,
        "isFiducialECALTrack": True, "hp_nValidPixelHits": 4,
        "hp_nValidHits": 4, "missingInnerHits": 0, "missingMiddleHits": 0,
        "missingOuterHits": 3, "pfRelIso03_chg": 0.01, "dRMinJet": 1.0,
        "dRMinElectron": 1.0, "dRMinMuon": 1.0, "dRMinTauHad": 1.0,
        "hp_trackerLayersWithMeasurement": 4, "caloEnergy": 1.0,
    }
    tracks = ak.Array([[
        {**base, "isHighPurityTrack": False},
        {**base, "isHighPurityTrack": True},
        {
            **base,
            "hp_trackerLayersWithMeasurement": 5,
            "isHighPurityTrack": False,
        },
    ]])

    nominal = fake_track_no_d0_mask(tracks, layer="NLayers4")
    study = fake_track_no_d0_mask(
        tracks, layer="combinedBins", require_high_purity=False
    )
    assert ak.to_list(nominal) == [[False, True, False]]
    assert ak.to_list(study) == [[True, True, True]]


def test_fake_track_dedx_max_over_median_applies_to_nlayers4_and_nlayers5():
    base = {
        "pt": 100.0, "eta": 0.8, "dxy": 0.1, "dz": 0.1,
        "inECALCrack": False, "inDTWheelGap": False,
        "inCSCTransition": False, "inTOBCrack": False,
        "isFiducialECALTrack": True, "hp_nValidPixelHits": 4,
        "hp_nValidHits": 4, "missingInnerHits": 0, "missingMiddleHits": 0,
        "missingOuterHits": 3, "pfRelIso03_chg": 0.01, "dRMinJet": 1.0,
        "dRMinElectron": 1.0, "dRMinMuon": 1.0, "dRMinTauHad": 1.0,
        "caloEnergy": 1.0, "isHighPurityTrack": True,
    }
    tracks = ak.Array([[
        {
            **base,
            "hp_trackerLayersWithMeasurement": 4,
            "dEdxMaximumOverMedian": 2.5,
        },
        {
            **base,
            "hp_trackerLayersWithMeasurement": 4,
            "dEdxMaximumOverMedian": 2.6,
        },
        {
            **base,
            "hp_trackerLayersWithMeasurement": 5,
            "dEdxMaximumOverMedian": 2.5,
        },
        {
            **base,
            "hp_trackerLayersWithMeasurement": 5,
            "dEdxMaximumOverMedian": 2.6,
        },
        {
            **base,
            "hp_trackerLayersWithMeasurement": 6,
            "dEdxMaximumOverMedian": 2.6,
        },
    ]])

    nlayers4 = fake_track_no_d0_mask(tracks, layer="NLayers4")
    nlayers5 = fake_track_no_d0_mask(tracks, layer="NLayers5")
    nlayers6plus = fake_track_no_d0_mask(tracks, layer="NLayers6plus")
    without_dedx_cut = fake_track_no_d0_mask(
        tracks, layer="NLayers4", require_dedx_max_over_median=False
    )

    assert ak.to_list(nlayers4) == [[True, False, False, False, False]]
    assert ak.to_list(nlayers5) == [[False, False, True, False, False]]
    assert ak.to_list(nlayers6plus) == [[False, False, False, False, True]]
    assert ak.to_list(without_dedx_cut) == [[True, True, False, False, False]]


def test_fake_track_dedx_max_over_median_skipped_when_field_absent():
    base = {
        "pt": 100.0, "eta": 0.8, "dxy": 0.1, "dz": 0.1,
        "inECALCrack": False, "inDTWheelGap": False,
        "inCSCTransition": False, "inTOBCrack": False,
        "isFiducialECALTrack": True, "hp_nValidPixelHits": 4,
        "hp_nValidHits": 4, "missingInnerHits": 0, "missingMiddleHits": 0,
        "missingOuterHits": 3, "pfRelIso03_chg": 0.01, "dRMinJet": 1.0,
        "dRMinElectron": 1.0, "dRMinMuon": 1.0, "dRMinTauHad": 1.0,
        "caloEnergy": 1.0, "isHighPurityTrack": True,
        "hp_trackerLayersWithMeasurement": 4,
    }
    tracks = ak.Array([[base]])

    assert ak.to_list(fake_track_no_d0_mask(tracks, layer="NLayers4")) == [[True]]


def test_fake_track_base_and_layer_cut_recombine_to_no_d0_mask():
    """fake_track_no_d0_mask must equal base_mask & layer_cut for every layer.

    Guards the split introduced so a caller looping over layer bins can
    compute the layer-independent base once instead of recomputing the full
    chain per bin -- the combinator wrapper and the split-out pieces must
    never drift apart.
    """
    base = {
        "pt": 100.0, "eta": 0.8, "dxy": 0.1, "dz": 0.1,
        "inECALCrack": False, "inDTWheelGap": False,
        "inCSCTransition": False, "inTOBCrack": False,
        "isFiducialECALTrack": True, "hp_nValidPixelHits": 4,
        "hp_nValidHits": 4, "missingInnerHits": 0, "missingMiddleHits": 0,
        "missingOuterHits": 3, "pfRelIso03_chg": 0.01, "dRMinJet": 1.0,
        "dRMinElectron": 1.0, "dRMinMuon": 1.0, "dRMinTauHad": 1.0,
        "caloEnergy": 1.0,
    }
    tracks = ak.Array([[
        {
            **base,
            "hp_trackerLayersWithMeasurement": 4,
            "isHighPurityTrack": True,
            "dEdxMaximumOverMedian": 2.4,
        },
        {
            **base,
            "hp_trackerLayersWithMeasurement": 4,
            "isHighPurityTrack": True,
            "dEdxMaximumOverMedian": 2.6,
        },
        {
            **base,
            "hp_trackerLayersWithMeasurement": 5,
            "isHighPurityTrack": False,
            "dEdxMaximumOverMedian": 2.4,
        },
        {
            **base,
            "hp_trackerLayersWithMeasurement": 6,
            "isHighPurityTrack": True,
            "dEdxMaximumOverMedian": 2.4,
        },
    ]])

    for layer in ("NLayers4", "NLayers5", "NLayers6plus", "combinedBins"):
        for d0_region in ("signal", "sideband"):
            for require_dedx in (True, False):
                combined = fake_track_base_mask(tracks, d0_region=d0_region) & (
                    fake_track_layer_cut(
                        tracks,
                        layer=layer,
                        require_dedx_max_over_median=require_dedx,
                    )
                )
                nominal = fake_track_no_d0_mask(
                    tracks,
                    layer=layer,
                    d0_region=d0_region,
                    require_dedx_max_over_median=require_dedx,
                )
                assert ak.to_list(combined) == ak.to_list(nominal), (
                    layer, d0_region, require_dedx
                )


def test_signal_track_comparison_applies_high_purity_to_every_layer_bin():
    base = {
        "pt": 100.0, "eta": 0.8, "dxy": 0.01, "dz": 0.1,
        "inECALCrack": False, "inDTWheelGap": False,
        "inCSCTransition": False, "inTOBCrack": False,
        "isFiducialECALTrack": True, "hp_nValidPixelHits": 4,
        "hp_nValidHits": 4, "missingInnerHits": 0, "missingMiddleHits": 0,
        "missingOuterHits": 3, "pfRelIso03_chg": 0.01, "dRMinJet": 1.0,
        "dRMinElectron": 1.0, "dRMinMuon": 1.0, "dRMinTauHad": 1.0,
        "caloEnergy": 1.0,
    }
    tracks = ak.Array([[
        {
            **base,
            "hp_trackerLayersWithMeasurement": 4,
            "isHighPurityTrack": False,
        },
        {
            **base,
            "hp_trackerLayersWithMeasurement": 5,
            "isHighPurityTrack": False,
        },
    ]])

    nominal = search_track_mask(tracks)
    without_high_purity = search_track_mask(
        tracks,
        require_high_purity=False,
    )

    assert ak.to_list(nominal) == [[False, False]]
    assert ak.to_list(without_high_purity) == [[True, True]]

    nominal_cutflow = search_track_cutflow_masks(tracks)
    comparison_cutflow = search_track_cutflow_masks(
        tracks,
        require_high_purity=False,
    )
    assert ak.to_list(nominal_cutflow["track_layers4plus"]) == [[True, True]]
    assert ak.to_list(nominal_cutflow["track_highPurity"]) == [[False, False]]
    assert ak.to_list(comparison_cutflow["track_highPurity"]) == [[True, True]]


def test_signal_cartesian_axes_reproduce_post_high_purity_cutflows():
    base = {
        "pt": 100.0, "eta": 0.8, "dxy": 0.01, "dz": 0.1,
        "inECALCrack": False, "inDTWheelGap": False,
        "inCSCTransition": False, "inTOBCrack": False,
        "isFiducialECALTrack": True, "hp_nValidPixelHits": 4,
        "hp_nValidHits": 4, "missingInnerHits": 0, "missingMiddleHits": 0,
        "missingOuterHits": 3, "pfRelIso03_chg": 0.01, "dRMinJet": 1.0,
        "dRMinElectron": 1.0, "dRMinMuon": 1.0, "dRMinTauHad": 1.0,
        "caloEnergy": 1.0,
    }
    tracks = ak.Array([[
        {**base, "hp_trackerLayersWithMeasurement": 4, "isHighPurityTrack": False},
        {**base, "hp_trackerLayersWithMeasurement": 4, "isHighPurityTrack": True},
        {**base, "hp_trackerLayersWithMeasurement": 5, "isHighPurityTrack": False},
        {**base, "hp_trackerLayersWithMeasurement": 6, "isHighPurityTrack": False},
    ]])
    generic = search_track_cutflow_masks(
        tracks,
        layer="combinedBins",
        require_high_purity=False,
    )
    post_high_purity_fields = list(generic)[
        list(generic).index("track_highPurity"):
    ]

    for layer in ("NLayers4", "NLayers5", "NLayers6plus", "combinedBins"):
        layer_axis = layer_mask(tracks, layer)
        for require_high_purity in (False, True):
            variant_axis = (
                tracks.isHighPurityTrack
                if require_high_purity
                else ak.ones_like(tracks.pt, dtype=bool)
            )
            expected = search_track_cutflow_masks(
                tracks,
                layer=layer,
                require_high_purity=require_high_purity,
            )
            for field in post_high_purity_fields:
                factored = generic[field] & layer_axis & variant_axis
                assert ak.to_list(factored) == ak.to_list(expected[field])


def test_met_no_mu_minus_muon_does_not_add_muon_twice():
    events = ak.Array([{"MetNoMu": {"pt": 100.0, "phi": 0.0}}])
    muons = ak.Array([[{"pt": 40.0, "phi": 0.0}]])

    pt, phi = met_no_mu_minus_lepton(events, muons, flavor="muon")

    assert ak.to_list(pt) == pytest.approx([100.0])
    assert ak.to_list(phi) == pytest.approx([0.0])


def test_met_no_mu_minus_electron_adds_visible_tag():
    events = ak.Array([{"MetNoMu": {"pt": 100.0, "phi": 0.0}}])
    electrons = ak.Array([[{"pt": 40.0, "phi": 0.0}]])

    pt, phi = met_no_mu_minus_lepton(events, electrons, flavor="electron")

    assert ak.to_list(pt) == pytest.approx([140.0])
    assert ak.to_list(phi) == pytest.approx([0.0])


def test_met_no_mu_minus_tau_adds_visible_tau():
    events = ak.Array([{"MetNoMu": {"pt": 100.0, "phi": 0.0}}])
    taus = ak.Array([[{"pt": 50.0, "phi": 0.0}]])

    pt, phi = met_no_mu_minus_lepton(events, taus, flavor="tau")

    assert ak.to_list(pt) == pytest.approx([150.0])
    assert ak.to_list(phi) == pytest.approx([0.0])


def test_table27_tau_control_mask_uses_pt_eta_and_deeptau_raw_working_points():
    taus = ak.Array(
        [[
            {
                "pt": 60.0,
                "eta": 1.0,
                "idDecayModeNewDMs": True,
                "rawDeepTau2018v2p5VSjet": 0.90,
                "rawDeepTau2018v2p5VSe": 0.20,
                "rawDeepTau2018v2p5VSmu": 0.40,
            },
            {
                "pt": 49.0,
                "eta": 1.0,
                "idDecayModeNewDMs": True,
                "rawDeepTau2018v2p5VSjet": 0.90,
                "rawDeepTau2018v2p5VSe": 0.20,
                "rawDeepTau2018v2p5VSmu": 0.40,
            },
            {
                "pt": 60.0,
                "eta": 1.0,
                "idDecayModeNewDMs": True,
                "rawDeepTau2018v2p5VSjet": 0.80,
                "rawDeepTau2018v2p5VSe": 0.20,
                "rawDeepTau2018v2p5VSmu": 0.40,
            },
        ]]
    )

    assert ak.to_list(hadronic_tau_control_object_mask(taus)) == [
        [True, False, False]
    ]


def test_table27_tau_control_mask_accepts_osunano_wp_ordinals():
    taus = ak.Array(
        [[
            {
                "pt": 60.0,
                "eta": 1.0,
                "idDecayModeNewDMs": True,
                "idDeepTau2018v2p5VSjet": 6,
                "idDeepTau2018v2p5VSe": 1,
                "idDeepTau2018v2p5VSmu": 1,
            },
            {
                "pt": 60.0,
                "eta": 1.0,
                "idDecayModeNewDMs": True,
                "idDeepTau2018v2p5VSjet": 5,
                "idDeepTau2018v2p5VSe": 1,
                "idDeepTau2018v2p5VSmu": 1,
            },
        ]]
    )

    assert ak.to_list(hadronic_tau_control_object_mask(taus)) == [[True, False]]


def test_lepton_pairs_keep_probe_coordinates_for_fiducial_maps():
    tags = ak.Array(
        [
            [
                {
                    "pt": 50.0,
                    "eta": 0.0,
                    "phi": 0.0,
                    "charge": 1,
                }
            ]
        ]
    )
    probes = ak.Array(
        [
            [
                {
                    "pt": 45.0,
                    "eta": 1.25,
                    "phi": -2.5,
                    "charge": -1,
                    "dRMinElectron": 0.2,
                    "dRMinVetoElectron": 0.1,
                    "dRMinMuon": 0.3,
                    "dRMinTauHad": 0.4,
                    "dRMinJet": 0.6,
                    "caloEnergy": 5.0,
                    "missingOuterHits": 3,
                    "hp_trackerLayersWithMeasurement": 4,
                }
            ]
        ]
    )

    pairs = build_lepton_veto_tag_probe_pairs(
        tags,
        probes,
        tag_mass=0.000511,
        probe_mass=0.000511,
    )

    assert ak.to_list(pairs.probe_pt) == [[45.0]]
    assert ak.to_list(pairs.probe_eta) == [[1.25]]
    assert ak.to_list(pairs.probe_phi) == [[-2.5]]
    assert ak.to_list(pairs.probe_passElectronVeto) == [[True]]
    assert ak.to_list(pairs.probe_passVetoElectronVeto) == [[False]]


def test_muon_pairs_keep_loose_muon_veto_separate_from_generic_veto():
    tags = ak.Array(
        [
            [
                {
                    "pt": 50.0,
                    "eta": 0.0,
                    "phi": 0.0,
                    "charge": 1,
                }
            ]
        ]
    )
    probes = ak.Array(
        [
            [
                {
                    "pt": 45.0,
                    "eta": 1.25,
                    "phi": -2.5,
                    "charge": -1,
                    "dRMinMuon": 0.1,
                    "dRMinLooseMuon": 0.3,
                    "caloEnergy": 5.0,
                    "missingOuterHits": 3,
                    "hp_trackerLayersWithMeasurement": 4,
                }
            ]
        ]
    )

    pairs = build_muon_veto_tag_probe_pairs(tags, probes)

    assert ak.to_list(pairs.probe_passMuonVeto) == [[False]]
    assert ak.to_list(pairs.probe_passLooseMuonVeto) == [[True]]


def test_fiducial_map_probe_uses_legacy_old_hit_cuts():
    tracks = ak.Array(
        [
            [
                {
                    "pt": 45.0,
                    "eta": 0.6,
                    "phi": 1.0,
                    "inECALCrack": False,
                    "inDTWheelGap": False,
                    "inCSCTransition": False,
                    "inTOBCrack": False,
                    "isFiducialECALTrack": True,
                    "hp_nValidPixelHits": 3,
                    "hp_nValidHits": 7,
                    "missingInnerHits": 0,
                    "missingMiddleHits": 0,
                    "pfRelIso03_chg": 0.01,
                    "dxy": 0.01,
                    "dz": 0.1,
                    "hp_trackerLayersWithMeasurement": 4,
                    "dRMinJet": 0.6,
                    "dRMinElectron": 0.2,
                    "dRMinMuon": 0.2,
                    "dRMinTauHad": 0.2,
                    "caloEnergy": 5.0,
                }
            ]
        ]
    )

    assert ak.to_list(fiducial_map_probe_track_mask(tracks, flavor="muon")) == [
        [True]
    ]
    assert ak.to_list(muon_veto_probe_track_mask(tracks)) == [[False]]


def test_lepton_background_probe_masks_apply_dedx_cut_after_dz():
    passing = {**PROBE_TRACK_BASE_FIELDS, "dEdxMaximumOverMedian": 2.5}
    failing = {**PROBE_TRACK_BASE_FIELDS, "dEdxMaximumOverMedian": 2.6}
    tracks = ak.Array([[passing, failing]])

    assert ak.to_list(muon_veto_probe_track_mask(tracks, layer="NLayers4")) == [
        [True, False]
    ]
    assert ak.to_list(
        lepton_veto_probe_track_mask(
            tracks, measured_veto="electron", layer="NLayers4"
        )
    ) == [[True, False]]
    assert ak.to_list(tau_veto_probe_track_mask(tracks, layer="NLayers4")) == [
        [True, False]
    ]


def test_lepton_background_probe_masks_apply_dedx_cut_within_combined_bins():
    """The dE/dx cut must fire for NLayers4/NLayers5 tracks even at the
    default ``layer="combinedBins"`` -- the layer bin every production call
    site actually uses. `MuonVetoProbeTrack`/etc. are built as ONE mixed-
    NLayers collection, not one collection per layer bin, so the dE/dx term
    cannot just look up a single working point for the caller-requested
    layer (that's `dedx_max_over_median_mask`, used by the fake-track
    background, which does call per layer bin) -- it must look up each
    track's OWN measured layer count instead (`probe_track_dedx_mask`).
    """

    # `passesTOBDzOrLambda` is only read by the cutflow-mask functions below,
    # not by `PROBE_TRACK_BASE_FIELDS`'s usual `base_probe_track_mask`
    # consumers, but including it on every track here lets the same fixtures
    # exercise both.
    nlayers4_pass = {
        **PROBE_TRACK_BASE_FIELDS,
        "passesTOBDzOrLambda": True,
        "hp_trackerLayersWithMeasurement": 4,
        "dEdxMaximumOverMedian": 2.5,
    }
    nlayers4_fail = {
        **PROBE_TRACK_BASE_FIELDS,
        "passesTOBDzOrLambda": True,
        "hp_trackerLayersWithMeasurement": 4,
        "dEdxMaximumOverMedian": 2.6,
    }
    nlayers5_fail = {
        **PROBE_TRACK_BASE_FIELDS,
        "passesTOBDzOrLambda": True,
        "hp_trackerLayersWithMeasurement": 5,
        "dEdxMaximumOverMedian": 2.6,
    }
    nlayers6_bad_dedx = {
        **PROBE_TRACK_BASE_FIELDS,
        "passesTOBDzOrLambda": True,
        "hp_trackerLayersWithMeasurement": 6,
        "dEdxMaximumOverMedian": 99.0,
    }
    tracks = ak.Array([[nlayers4_pass, nlayers4_fail, nlayers5_fail, nlayers6_bad_dedx]])

    # No explicit `layer=` -- matches every production call site.
    expected = [True, False, False, True]
    assert ak.to_list(muon_veto_probe_track_mask(tracks)) == [expected]
    assert ak.to_list(
        lepton_veto_probe_track_mask(tracks, measured_veto="electron")
    ) == [expected]
    assert ak.to_list(tau_veto_probe_track_mask(tracks)) == [expected]

    muon_masks = muon_veto_probe_track_cutflow_masks(tracks)
    assert ak.to_list(muon_masks["track_dedx"]) == [expected]
    assert ak.to_list(muon_masks["track_layers4plus"]) == [expected]

    tau_masks = tau_veto_probe_track_cutflow_masks(tracks)
    assert ak.to_list(tau_masks["track_dedx"]) == [expected]
    assert ak.to_list(tau_masks["track_layers4plus"]) == [expected]


def test_probe_track_dedx_mask_is_per_track_not_per_collection():
    tracks = ak.Array([[
        {"hp_trackerLayersWithMeasurement": 4, "hp_nValidTrackerHits": 4, "dEdxMaximumOverMedian": 2.4},
        {"hp_trackerLayersWithMeasurement": 4, "hp_nValidTrackerHits": 4, "dEdxMaximumOverMedian": 2.6},
        {"hp_trackerLayersWithMeasurement": 5, "hp_nValidTrackerHits": 5, "dEdxMaximumOverMedian": 2.4},
        {"hp_trackerLayersWithMeasurement": 5, "hp_nValidTrackerHits": 5, "dEdxMaximumOverMedian": 2.6},
        {"hp_trackerLayersWithMeasurement": 7, "hp_nValidTrackerHits": 7, "dEdxMaximumOverMedian": 1000.0},
    ]])

    assert ak.to_list(probe_track_dedx_mask(tracks)) == [
        [True, False, True, False, True]
    ]


def test_fake_track_layer_cut_dedx_lookup_is_unaffected_by_probe_track_dedx_mask():
    """`fake_track_layer_cut` must keep using the single-layer-bin lookup
    (`dedx_max_over_median_mask`), not the new per-track lookup -- it already
    loops per layer bin itself, so a track collection passed to it for a
    given layer bin is evaluated against that bin's working point uniformly.
    Guards against `probe_track_dedx_mask` silently replacing it too.
    """

    base = {
        "pt": 100.0, "eta": 0.8, "dxy": 0.1, "dz": 0.1,
        "inECALCrack": False, "inDTWheelGap": False,
        "inCSCTransition": False, "inTOBCrack": False,
        "isFiducialECALTrack": True, "hp_nValidPixelHits": 4,
        "hp_nValidHits": 4, "hp_nValidTrackerHits": 6,
        "missingInnerHits": 0, "missingMiddleHits": 0,
        "missingOuterHits": 3, "pfRelIso03_chg": 0.01, "dRMinJet": 1.0,
        "dRMinElectron": 1.0, "dRMinMuon": 1.0, "dRMinTauHad": 1.0,
        "caloEnergy": 1.0, "isHighPurityTrack": True,
    }
    tracks = ak.Array([[
        {**base, "hp_trackerLayersWithMeasurement": 6, "dEdxMaximumOverMedian": 99.0},
    ]])

    # NLayers6plus has no configured working point -- must stay a no-op,
    # exactly as before this session's change.
    assert ak.to_list(fake_track_layer_cut(tracks, layer="NLayers6plus")) == [[True]]


def test_lepton_background_probe_masks_dedx_cut_can_be_disabled():
    tracks = ak.Array([[{**PROBE_TRACK_BASE_FIELDS, "dEdxMaximumOverMedian": 2.6}]])

    assert ak.to_list(
        muon_veto_probe_track_mask(
            tracks, layer="NLayers4", require_dedx_max_over_median=False
        )
    ) == [[True]]


def test_lepton_background_probe_masks_dedx_cut_skipped_when_field_absent():
    tracks = ak.Array([[PROBE_TRACK_BASE_FIELDS]])

    assert ak.to_list(muon_veto_probe_track_mask(tracks, layer="NLayers4")) == [[True]]
    assert ak.to_list(
        lepton_veto_probe_track_mask(
            tracks, measured_veto="electron", layer="NLayers4"
        )
    ) == [[True]]
    assert ak.to_list(tau_veto_probe_track_mask(tracks, layer="NLayers4")) == [[True]]


def test_base_probe_track_mask_dedx_cut_is_opt_in_and_does_not_leak_into_search_track_mask():
    """The dE/dx term defaults off on `base_probe_track_mask` itself.

    `search_track_mask` (the signal-region selection -- a separate,
    deliberately unsettled question from the lepton-background probe-track
    cuts, see the analysis's "adding highPurity to the signal selection"
    investigation) shares `base_probe_track_mask` and must not silently start
    applying a dE/dx cut just because a track carries
    `dEdxMaximumOverMedian` -- e.g. because a lepton-background probe-track
    mask was evaluated earlier in the same event pass and attached the field.
    """
    failing_dedx = {**PROBE_TRACK_BASE_FIELDS, "dEdxMaximumOverMedian": 2.6}
    tracks = ak.Array([[failing_dedx]])

    assert ak.to_list(muon_veto_probe_track_mask(tracks, layer="NLayers4")) == [
        [False]
    ]
    assert ak.to_list(base_probe_track_mask(tracks, layer="NLayers4")) == [[True]]
    assert ak.to_list(search_track_mask(tracks, layer="NLayers4")) == [[True]]


def _table16_probe_fields(*, is_high_purity, dedx_max_over_median):
    return {
        "pt": 100.0, "eta": 0.8, "dxy": 0.01, "dz": 0.1,
        "inECALCrack": False, "inDTWheelGap": False, "inCSCTransition": False,
        "inTOBCrack": False,
        "isFiducialECALTrack": True, "passesTOBDzOrLambda": True,
        "hp_nValidPixelHits": 4, "hp_nValidHits": 4, "hp_nValidTrackerHits": 4,
        "missingInnerHits": 0, "missingMiddleHits": 0,
        "pfRelIso03_chg": 0.01, "dRMinJet": 1.0,
        "dRMinElectron": 1.0, "dRMinTauHad": 1.0,
        "caloEnergy": 1.0, "hp_trackerLayersWithMeasurement": 4,
        "isHighPurityTrack": is_high_purity,
        "dEdxMaximumOverMedian": dedx_max_over_median,
    }


def test_muon_veto_probe_track_cutflow_masks_places_purity_and_dedx_after_dz():
    tracks = ak.Array([[
        _table16_probe_fields(is_high_purity=False, dedx_max_over_median=2.5),
        _table16_probe_fields(is_high_purity=True, dedx_max_over_median=2.6),
        _table16_probe_fields(is_high_purity=True, dedx_max_over_median=2.5),
    ]])

    masks = muon_veto_probe_track_cutflow_masks(tracks, layer="NLayers4")

    keys = list(masks.keys())
    assert keys.index("track_highPurity") == keys.index("track_dz0p5") + 1
    assert keys.index("track_dedx") == keys.index("track_highPurity") + 1
    assert ak.to_list(masks["track_dz0p5"]) == [[True, True, True]]
    assert ak.to_list(masks["track_highPurity"]) == [[False, True, True]]
    assert ak.to_list(masks["track_dedx"]) == [[False, False, True]]
    assert ak.to_list(masks["track_layers4plus"]) == [[False, False, True]]


def test_tau_veto_probe_track_cutflow_masks_places_purity_and_dedx_after_dz():
    def fields(*, is_high_purity, dedx_max_over_median):
        base = _table16_probe_fields(
            is_high_purity=is_high_purity, dedx_max_over_median=dedx_max_over_median
        )
        base["dRMinMuon"] = 1.0
        return base

    tracks = ak.Array([[
        fields(is_high_purity=False, dedx_max_over_median=2.5),
        fields(is_high_purity=True, dedx_max_over_median=2.6),
        fields(is_high_purity=True, dedx_max_over_median=2.5),
    ]])

    masks = tau_veto_probe_track_cutflow_masks(tracks, layer="NLayers4")

    keys = list(masks.keys())
    assert keys.index("track_highPurity") == keys.index("track_dz0p5") + 1
    assert keys.index("track_dedx") == keys.index("track_highPurity") + 1
    assert ak.to_list(masks["track_dedx"]) == [[False, False, True]]
    assert ak.to_list(masks["track_layers4plus"]) == [[False, False, True]]


def test_fiducial_map_probe_track_mask_stays_independent_of_high_purity():
    tracks = ak.Array([[
        {
            "pt": 45.0, "eta": 0.6, "phi": 1.0,
            "inECALCrack": False, "inDTWheelGap": False,
            "inCSCTransition": False, "inTOBCrack": False,
            "isFiducialECALTrack": True, "hp_nValidPixelHits": 3,
            "hp_nValidHits": 7, "hp_nValidTrackerHits": 4,
            "missingInnerHits": 0, "missingMiddleHits": 0,
            "pfRelIso03_chg": 0.01, "dxy": 0.01, "dz": 0.1,
            "hp_trackerLayersWithMeasurement": 4,
            "dRMinJet": 0.6, "dRMinElectron": 0.2, "dRMinMuon": 0.2,
            "dRMinTauHad": 0.2, "caloEnergy": 5.0,
            "isHighPurityTrack": False,
            "dEdxMaximumOverMedian": 99.0,
        }
    ]])

    # Deliberately fails both high-purity and dE/dx working points, but the
    # fiducial-map probe is documented to measure detector hot spots
    # independent of track quality -- it must still pass.
    assert ak.to_list(fiducial_map_probe_track_mask(tracks, flavor="muon")) == [
        [True]
    ]
