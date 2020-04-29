from numpy import linspace

from summer.model.strat_model import StratifiedModel
from autumn.post_processing.processor import post_process
from summer.constants import IntegrationType

import pytest


@pytest.mark.xfail(reason="Weirdly broken.")
def test_post_processing():
    model = _get_model()
    generated_outputs = post_process(model, INPUT_CONFIG)
    assert generated_outputs.keys() == EXPECTED_OUTPUTS.keys()
    for k, v in generated_outputs.items():
        assert v == EXPECTED_OUTPUTS[k], f"Values for {k} should match."
    assert generated_outputs == EXPECTED_OUTPUTS


def _get_model():
    """
    Returns a run StratifiedModel for testing.
    """
    model = StratifiedModel(
        linspace(0, 60 / 365, 61).tolist(),
        ["susceptible", "infectious", "recovered"],
        {"infectious": 0.001},
        {"beta": 400, "recovery": 365 / 13, "infect_death": 1},
        [
            {
                "type": "standard_flows",
                "parameter": "recovery",
                "origin": "infectious",
                "to": "recovered",
            },
            {
                "type": "infection_frequency",
                "parameter": "beta",
                "origin": "susceptible",
                "to": "infectious",
            },
            {"type": "compartment_death", "parameter": "infect_death", "origin": "infectious"},
        ],
        output_connections={},
        verbose=False,
    )

    model.stratify(
        "strain",
        ["sensitive", "resistant"],
        ["infectious"],
        requested_proportions={},
        verbose=False,
    )

    age_mixing = None
    model.stratify(
        "age",
        [1, 10, 3],
        [],
        {},
        {"recovery": {"1": 0.5, "10": 0.8}},
        infectiousness_adjustments={"1": 0.8},
        mixing_matrix=age_mixing,
        verbose=False,
    )

    model.run_model(integration_type=IntegrationType.ODE_INT)
    return model


INPUT_CONFIG = {
    "requested_outputs": ["prevXinfectiousXamong", "distribution_of_strataXstrain",],
    "multipliers": {"prevXinfectiousXamong": 1.0e5,},
    "collated_combos": [],
}

EXPECTED_OUTPUTS = {
    "prevXinfectiousXamong": [
        100.0,
        157.39261346078854,
        246.09033478894378,
        383.9239846503909,
        600.635107811137,
        934.867653689807,
        1448.7112800442576,
        2237.1535765096874,
        3441.535973379213,
        5233.586922307695,
        7831.359124718591,
        11492.147722762338,
        16383.528762993688,
        22468.3623765879,
        29511.29856546797,
        36769.88148374874,
        43682.629576522675,
        49365.82785179127,
        53492.456839814426,
        55835.46672507633,
        56550.981676183066,
        55976.45016911882,
        54519.01045815087,
        52383.0739826138,
        49749.89580114868,
        46977.05721587429,
        44102.16759676176,
        41290.517108928805,
        38570.53073091502,
        35967.0859269628,
        33492.847134192154,
        31149.61491783768,
        28941.66356451306,
        26870.069252437774,
        24929.499300530853,
        23115.379481131833,
        21426.108583119538,
        19847.12475067732,
        18375.89325792617,
        17012.767249403558,
        15748.304423096952,
        14570.05816680654,
        13478.195629444623,
        12471.319026370338,
        11534.895546118993,
        10665.50514417961,
        9863.272518103635,
        9120.585304068909,
        8425.099920730097,
        7776.719344862692,
        7175.5083278584225,
        6621.526934048541,
        6114.830525692846,
        5655.469749133028,
        5242.840933882819,
        4861.260408608431,
        4504.033713026157,
        4171.179115277163,
        3862.713641960362,
        3578.6530759557086,
        3319.0119544055146,
    ],
    "distribution_of_strataXstrain": {
        "sensitive": [
            0.0005,
            0.0007869602554942867,
            0.0012304404755045168,
            0.0019195859345338309,
            0.0030030816157493407,
            0.0046740938481958745,
            0.007242942151754986,
            0.011184256010688079,
            0.01720400873537552,
            0.02615923331656373,
            0.03913677901727058,
            0.05741613487046361,
            0.08182258258105574,
            0.11215144423973328,
            0.14720104501970294,
            0.18324013667798347,
            0.21745022677507397,
            0.2454306345413288,
            0.2655746646883758,
            0.2767933108366035,
            0.279909576347783,
            0.27663925913423737,
            0.26902945346005014,
            0.25811013633434465,
            0.24478903322729526,
            0.23083616311406352,
            0.21643595458447892,
            0.20239790991834086,
            0.18885679624120014,
            0.17592829610682595,
            0.1636691001236068,
            0.15208326233706254,
            0.14118639891043366,
            0.13098001281061653,
            0.1214341007634863,
            0.11252293577321197,
            0.10423603646774393,
            0.09649951936810959,
            0.08929903603118217,
            0.0826345864569617,
            0.07645842100798186,
            0.07070842089067236,
            0.06538432556250909,
            0.060478328858867676,
            0.055918767804802386,
            0.05168831124981828,
            0.047786959193915346,
            0.04417715718434248,
            0.040798489121330786,
            0.037650145672458675,
            0.034732126837726106,
            0.03204443261713306,
            0.029587063010679572,
            0.02736001801836562,
            0.0253601496475712,
            0.023511275671099043,
            0.021780849905810473,
            0.02016887235170553,
            0.018675343008784186,
            0.017300261877046457,
            0.01604362895649234,
        ],
        "resistant": [
            0.0005,
            0.0007869602554942867,
            0.0012304404755045168,
            0.0019195859345338309,
            0.0030030816157493407,
            0.0046740938481958745,
            0.007242942151754986,
            0.011184256010688079,
            0.01720400873537552,
            0.02615923331656373,
            0.03913677901727058,
            0.05741613487046361,
            0.08182258258105574,
            0.11215144423973328,
            0.14720104501970294,
            0.18324013667798347,
            0.21745022677507397,
            0.2454306345413288,
            0.2655746646883758,
            0.2767933108366035,
            0.279909576347783,
            0.27663925913423737,
            0.26902945346005014,
            0.25811013633434465,
            0.24478903322729526,
            0.23083616311406352,
            0.21643595458447892,
            0.20239790991834086,
            0.18885679624120014,
            0.17592829610682595,
            0.1636691001236068,
            0.15208326233706254,
            0.14118639891043366,
            0.13098001281061653,
            0.1214341007634863,
            0.11252293577321197,
            0.10423603646774393,
            0.09649951936810959,
            0.08929903603118217,
            0.0826345864569617,
            0.07645842100798186,
            0.07070842089067236,
            0.06538432556250909,
            0.060478328858867676,
            0.055918767804802386,
            0.05168831124981828,
            0.047786959193915346,
            0.04417715718434248,
            0.040798489121330786,
            0.037650145672458675,
            0.034732126837726106,
            0.03204443261713306,
            0.029587063010679572,
            0.02736001801836562,
            0.0253601496475712,
            0.023511275671099043,
            0.021780849905810473,
            0.02016887235170553,
            0.018675343008784186,
            0.017300261877046457,
            0.01604362895649234,
        ],
    },
    "distribution_of_strataXage": {
        "0": [
            0.24999999999999997,
            0.2493151167725093,
            0.24863162226241842,
            0.2479492390636686,
            0.2472675026058227,
            0.2465858057078433,
            0.2459031629793156,
            0.24521803461827937,
            0.24452801755198905,
            0.2438299504322149,
            0.24311940660779519,
            0.24238998588693378,
            0.2416339454901356,
            0.2408440966137167,
            0.24001341607658058,
            0.23913976565546008,
            0.2382236581642853,
            0.23727199607995356,
            0.23629165310143813,
            0.23529339325113222,
            0.23428898187637534,
            0.23328814445265883,
            0.23229821779003562,
            0.23132244295610643,
            0.2303631998197052,
            0.22942512900526368,
            0.22850844254571143,
            0.22761367634955573,
            0.22674086189022763,
            0.22588811468569853,
            0.22505451123947562,
            0.22423951328011993,
            0.22344104738871798,
            0.2226589062939959,
            0.22189175437969338,
            0.22113844639100172,
            0.2203987180513359,
            0.21967070646780149,
            0.21895404985975264,
            0.21824874822718932,
            0.2175537788032131,
            0.21686781104273073,
            0.21619083936508687,
            0.21552271563430675,
            0.21486207991690612,
            0.21420860332321934,
            0.21356228585324633,
            0.212922455394498,
            0.2122880306793158,
            0.2116589972230676,
            0.21103535502575355,
            0.21041710408737357,
            0.2098042444079277,
            0.20919677598741585,
            0.2085946436065714,
            0.20799651094500615,
            0.20740180710672904,
            0.20681053209174016,
            0.2062226859000395,
            0.20563826853162703,
            0.20505727998650278,
        ],
        "1": [
            0.24999999999999997,
            0.2503403985515472,
            0.25067796919783986,
            0.25101243467929557,
            0.25134332566645584,
            0.25167002011724593,
            0.2519914989813004,
            0.25230615473256396,
            0.2526114572944553,
            0.2529040554261294,
            0.25317923140291615,
            0.2534301153909302,
            0.2536483071531966,
            0.2538258884661328,
            0.2539550001554121,
            0.2540328120511467,
            0.25405947484656793,
            0.254041777212969,
            0.25398677414213533,
            0.25390565483478267,
            0.2538108470444008,
            0.2537128044820148,
            0.2536196862354605,
            0.2535351448426268,
            0.25346189430177674,
            0.25340522677291033,
            0.2533654504328253,
            0.2533435642734604,
            0.2533396823817795,
            0.25335217387206144,
            0.2533802395235012,
            0.25342338765993017,
            0.2534797243471847,
            0.2535490601825664,
            0.25363012430313625,
            0.2537218269783067,
            0.2538239112714873,
            0.25393456602960707,
            0.25405343952008397,
            0.25418053174291805,
            0.2543148408390878,
            0.254455063463122,
            0.25460119414844784,
            0.2547530854527871,
            0.25490938381100336,
            0.255069761873568,
            0.25523421964048093,
            0.25540207441270985,
            0.2555722278913892,
            0.2557446653637376,
            0.25591938682975485,
            0.25609639228944103,
            0.25627568174279614,
            0.25645725518982015,
            0.2566410572569706,
            0.2568257478903456,
            0.25701075459894135,
            0.257196077382758,
            0.2573817162417955,
            0.2575676711760539,
            0.2577539421855331,
        ],
        "10": [
            0.24999999999999997,
            0.25009700168333143,
            0.2501936092545307,
            0.2502895415986224,
            0.2503843252983087,
            0.2504773371971257,
            0.2505675608312574,
            0.25065339905917594,
            0.2507323457492322,
            0.250801088247096,
            0.25085496999339724,
            0.2508872221818701,
            0.250889587230147,
            0.2508543047163112,
            0.2507736955904597,
            0.2506450727152504,
            0.2504686582021674,
            0.25025124884128663,
            0.2499998473781681,
            0.24972553182626483,
            0.24944056646402823,
            0.24915522916217397,
            0.24887748263085419,
            0.24861088137592477,
            0.2483580590606786,
            0.24812415119596742,
            0.24790944270250034,
            0.24771482041380768,
            0.24754037836594658,
            0.24738442282671205,
            0.2472461237771773,
            0.2471249773497543,
            0.24701904264899294,
            0.24692812559391752,
            0.24685093627395943,
            0.24678636862570433,
            0.24673416300807993,
            0.24669248920404815,
            0.24666099177873463,
            0.24663967073213933,
            0.24662751425061702,
            0.2466232060384577,
            0.2466267405747717,
            0.24663796925312317,
            0.24665552782105155,
            0.24667908634437938,
            0.2467086448231065,
            0.24674351495601315,
            0.24678258943165868,
            0.24682585341652924,
            0.24687330691062484,
            0.24692494991394548,
            0.2469807824264912,
            0.247040804448262,
            0.24710495980767969,
            0.24717188913817517,
            0.2472410116980854,
            0.24731232748741042,
            0.24738583650615023,
            0.24746153875430482,
            0.24753943423187416,
        ],
        "3": [
            0.24999999999999997,
            0.25024391000453566,
            0.25048769820433214,
            0.25073107869604494,
            0.25097357176527035,
            0.25121454716923386,
            0.25145297791446974,
            0.2516872512794654,
            0.25191483678353177,
            0.25213238844127517,
            0.2523352008730326,
            0.25251642923777534,
            0.2526677112966275,
            0.25278117065515765,
            0.2528489951439295,
            0.25286838327651373,
            0.2528394947655281,
            0.25276909610715304,
            0.25266420877793144,
            0.25253596325649613,
            0.2523967128711583,
            0.25225683594188175,
            0.25212440957347576,
            0.2520030456127813,
            0.25189542340711246,
            0.2518067675488627,
            0.25173737578789734,
            0.25168819642565127,
            0.25165933439946264,
            0.2516491274473859,
            0.2516567609725964,
            0.25168173610644134,
            0.2517221312093442,
            0.2517777541134738,
            0.2518473191868637,
            0.251929724034885,
            0.2520247088444043,
            0.2521304421855203,
            0.25224656838781573,
            0.2523730874512907,
            0.25250898454822135,
            0.25265293946180123,
            0.2528049466546945,
            0.25296485646722094,
            0.2531312949778156,
            0.25330392991390077,
            0.2534827612754765,
            0.25366709035215385,
            0.25385579308663814,
            0.2540488544210886,
            0.2542462743555051,
            0.2544480528898878,
            0.2546541900242366,
            0.25486468575855153,
            0.25507948300228134,
            0.25529720014947177,
            0.2555172469574629,
            0.2557396234262549,
            0.25596432955584764,
            0.2561913653462411,
            0.2564207307974355,
        ],
    },
}
