"""
Generate latex/shared/generated_macros.tex from outputs/stats/summary_stats.json.

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
    ('Reproducibility & hyperparameters', ['Seed', 'ValSizeFrac', 'ValSizePct',
                                 'StructuralC', 'StructuralMaxIter', 'TfidfClfC', 'TfidfClfMaxIter',
                                 'PosNEstimators', 'PosClassWeight', 'EmbeddingC', 'EmbeddingMaxIter',
                                 'SvmC', 'SvmGamma',
                                 'TfidfMaxFeatures', 'TfidfSublinearTf', 'TfidfMinDf', 'EmbeddingModelName']),
    ('Cascade thresholds',     ['TauOneDefault', 'TauTwoDefault']),
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
    ('Threshold ablation',     ['AblGridPoints', 'AblTierOneLowFone', 'AblTierOneHighFone',
                                 'AblTierOneLowPct', 'AblTierOneBreakPct', 'AblTierOneBreakPctHigh',
                                 'AblBestFone', 'AblDefaultFone', 'AblHighTauOneLowFone', 'AblTauTwoMaxTierThreePct',
                                 'AblTThreeRateZeroFive', 'AblTThreeRateZeroSix', 'AblTThreeRateZeroSeven',
                                 'AblTThreeRateZeroEight', 'AblTThreeRateZeroNine', 'AblTThreeRateZeroNineFive',
                                 'AblTThreeRateZeroNineNine', 'AblTThreeRateZeroNineNineNine',
                                 'AblTThreeRateZeroNineNineNineNine']),
    ('SVM baseline',           ['SvmFone', 'SvmAuc', 'SvmCsFone', 'SvmLatency']),
    ('Kaggle scores',          ['KaggleBaselinePublic', 'KaggleBaselinePrivate']),
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
            if key in macros and not key.startswith('NTwoV'):
                lines.append(f'\\newcommand{{\\{key}}}{{{macros[key]}}}')
                emitted.add(key)
        lines.append('')

    # Catch any computed keys not in the explicit group list
    remaining = {k: v for k, v in macros.items()
                 if k not in emitted and not k.startswith('NTwoV')}
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
