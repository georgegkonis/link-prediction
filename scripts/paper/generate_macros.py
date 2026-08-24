"""
Generate latex/shared/generated_macros.tex from outputs/stats/summary_stats.json.

All computation lives in scripts/paper/compute_summary_stats.py — this script only formats the
already-computed macros dict into \\newcommand definitions, grouped for readability. It never
touches data/raw/, data/interim/, or outputs/predictions/, so it runs from a clean checkout as long
as summary_stats.json (committed) is present.

Reads:
  outputs/stats/summary_stats.json

Writes:
  latex/shared/generated_macros.tex

Usage:
    python -m scripts.paper.generate_macros
"""

import json
import pathlib

from src.utils.log_utils import setup_logging

log = setup_logging('generate_macros')

STATS  = pathlib.Path('outputs/stats')
SHARED = pathlib.Path('latex/shared')

_GROUPS = [
    ('Dataset sizes',          ['TrainPairs', 'TrainPairsNoSelf', 'TrainSplitSize', 'ValSplitSize',
                                 'TestPairs', 'SvmSampleSize', 'GraphNodes', 'GraphMeanDegree',
                                 'GraphNodesTotal',
                                 'TrainPosCount', 'TrainNegCount', 'TrainPosPct', 'TrainNegPct']),
    ('Leakage audit',          ['LeakageExact', 'LeakageExactPct', 'LeakageReversed',
                                 'LeakageReversedPct', 'SelfLoopsTrain', 'SelfLoopsTrainPos',
                                 'SelfLoopsTrainNeg', 'SelfLoopsTest', 'IntraTrainDuplicates',
                                 'IntraTrainDuplicatesRows']),
    ('Separability thresholds',['CnThreshold', 'TfidfThreshold', 'StructCnMeanPos', 'StructCnMeanNeg',
                                 'StructTfidfMeanPos', 'StructTfidfMeanNeg',
                                 'GraphCoverageTrainPct', 'GraphCoverageTestPct']),
    ('Difficulty — train',     ['TrainDiffSelfLoopN', 'TrainDiffSelfLoopPct', 'TrainDiffHighCnN',
                                 'TrainDiffHighCnPct', 'TrainDiffHighTextsimN', 'TrainDiffHighTextsimPct',
                                 'TrainDiffHardN', 'TrainDiffHardPct', 'TrainDiffTrivialPct']),
    ('Difficulty — test',      ['TestDiffSelfLoopN', 'TestDiffSelfLoopPct', 'TestDiffHighCnN',
                                 'TestDiffHighTextsimN', 'TestDiffHighTextsimPct',
                                 'TestDiffHardN', 'TestDiffHardPct']),
    ('Cascade thresholds',     ['TauOneDefault', 'TauTwoDefault', 'NTwoVDim']),
    ('Cascade val results',    ['CascadeValFone', 'CascadeValAuc',
                                 'TierOneCount', 'TierOnePct', 'TierOneAcc', 'TierOneFone',
                                 'TierTwoCount', 'TierTwoPct', 'TierTwoAcc', 'TierTwoFone',
                                 'TierThreeCount', 'TierThreePct', 'TierThreeAcc', 'TierThreeFone',
                                 'TierOnePosPct', 'TierThreeTruePosPct', 'TierThreePredPosPct']),
    ('Tier x difficulty',      ['TierOneHardN', 'TierOneHardAcc', 'TierOneHighTextsimN', 'TierOneHighTextsimAcc',
                                 'TierOneHighCnN', 'TierOneHighCnAcc',
                                 'TierTwoHardN', 'TierTwoHardAcc', 'TierTwoHighTextsimN', 'TierTwoHighTextsimAcc',
                                 'TierThreeHardN', 'TierThreeHardAcc',
                                 'TierThreeHighTextsimN', 'TierThreeHighTextsimAcc']),
    ('Test-set tier dist',     ['TestTierOneCount', 'TestTierOnePct', 'TestTierZeroPct', 'TestTierTwoPct']),
    ('Cold-start',             ['ColdStartCount', 'ColdStartPct', 'ColdStartNonCount']),
    ('Hard residual',          ['HardResCount', 'HardResAcc', 'HardResMissingN', 'HardResMissingPct',
                                 'HardResCompleteN', 'HardResCompleteAcc', 'HardResIncompleteAcc',
                                 'HardResNodeId', 'HardResNodePairs', 'HardResNodePredPosPct']),
    ('Efficiency',             ['ThroughputSec', 'ThroughputRate', 'ThroughputLatency', 'TierThreeCallRateMax']),
    ('Error analysis',         ['CascadeTotalErrors', 'CascadeErrorsHardN', 'CascadeErrorsHardPct',
                                 'CascadeErrorsTrivialN', 'CascadeErrorsTrivialPct', 'CascadeHardAcc', 'CascadeHardFone',
                                 'SvmTotalErrors', 'CascadeVsSvmRatio', 'SvmErrorsHardN', 'SvmErrorsHardPct',
                                 'SvmErrorsTrivialN', 'SvmErrorsTrivialPct', 'SvmHighTextsimAcc', 'SvmHighTextsimFone',
                                 'SvmHardAcc', 'SvmHardFone']),
    ('Node2Vec ablation',      ['NTwoVWithCallPct', 'NTwoVWithCallN', 'NTwoVWithOverallFone',
                                 'NTwoVWithConfFone', 'NTwoVNoOverallFone', 'NTwoVNoConfFone',
                                 'NTwoVNoAccOverall', 'NTwoVWithAccOverall',
                                 'NTwoVIdOneCoverage', 'NTwoVIdTwoCoverage', 'NTwoVZeroVecPct',
                                 'NTwoVZeroVecN', 'NTwoVTestTierOnePct']),
    ('Threshold ablation',     ['AblGridPoints', 'AblTierOneLowFone', 'AblTierOneHighFone',
                                 'AblTierOneLowPct', 'AblTierOneBreakPct', 'AblTierOneBreakPctHigh',
                                 'AblBestFone', 'AblDefaultFone', 'AblHighTauOneLowFone', 'AblTauTwoMaxTierThreePct',
                                 'AblTThreeRateZeroFive', 'AblTThreeRateZeroSix', 'AblTThreeRateZeroSeven',
                                 'AblTThreeRateZeroEight', 'AblTThreeRateZeroNine']),
    ('SVM baseline',           ['SvmFone', 'SvmAuc', 'SvmCsFone', 'SvmLatency']),
    ('Kaggle scores',          ['KaggleBaselinePublic', 'KaggleBaselinePrivate',
                                 'NTwoVKagglePublic', 'NTwoVKagglePrivate']),
]


def _emit_tex(macros: dict[str, str]) -> None:
    lines = [
        '% AUTO-GENERATED by scripts/paper/generate_macros.py — do not edit by hand.',
        '% Re-run: make thesis-macros',
        '',
    ]
    emitted = set()
    for group_name, keys in _GROUPS:
        lines.append(f'%% {group_name}')
        for key in keys:
            if key in macros:
                lines.append(f'\\newcommand{{\\{key}}}{{{macros[key]}}}')
                emitted.add(key)
        lines.append('')

    # Catch any computed keys not in the explicit group list
    remaining = {k: v for k, v in macros.items() if k not in emitted}
    if remaining:
        lines.append('%% Additional computed macros')
        for k, v in sorted(remaining.items()):
            lines.append(f'\\newcommand{{\\{k}}}{{{v}}}')
        lines.append('')

    out = SHARED / 'generated_macros.tex'
    out.write_text('\n'.join(lines))
    log.info('Wrote %d macros → %s', len(macros), out)


def main():
    stats_path = STATS / 'summary_stats.json'
    if not stats_path.exists():
        raise SystemExit(
            f'{stats_path} not found — run `make compute-stats` first '
            '(requires the full local data/feature/train pipeline output).'
        )
    stats = json.loads(stats_path.read_text())
    _emit_tex(stats['macros'])


if __name__ == '__main__':
    main()
