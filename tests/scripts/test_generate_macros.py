"""Tests for the pure Greek-number formatters in scripts/thesis/compute_summary_stats.py.

Only the formatters are exercised — `_load`, `_compute_macros` and `_compute_figures` read
`data/`/`outputs/predictions/dsaa/`, so they are out of scope here.
"""

import pytest

from scripts.thesis.compute_summary_stats import gfloat, gint, gpct


@pytest.mark.parametrize('n, expected', [
    (0, '0'),
    (7, '7'),
    (999, '999'),
    (1000, '1.000'),
    (948232, '948.232'),
    (238365, '238.365'),
    (1234567, '1.234.567'),
])
def test_gint_uses_greek_thousands_separator(n, expected):
    assert gint(n) == expected


def test_gint_accepts_numpy_and_float_input():
    import numpy as np
    assert gint(np.int64(1000)) == '1.000'
    assert gint(1000.0) == '1.000'
    assert gint(1000.7) == '1.000'      # truncates, does not round


def test_gint_negative():
    assert gint(-1000) == '-1.000'


@pytest.mark.parametrize('x, d, expected', [
    (0.9986, 4, '0{,}9986'),
    (0.5, 4, '0{,}5000'),
    (0.5, 2, '0{,}50'),
    (1.0, 4, '1{,}0000'),
    (1234.5, 2, '1.234{,}50'),
    (1234567.891, 3, '1.234.567{,}891'),
    (0.0, 1, '0{,}0'),
])
def test_gfloat_decimal_comma_and_thousands(x, d, expected):
    assert gfloat(x, d) == expected


def test_gfloat_default_precision_is_four():
    assert gfloat(0.123456) == '0{,}1235'


def test_gfloat_rounds_half_even_like_format():
    assert gfloat(0.99999, 4) == '1{,}0000'


@pytest.mark.parametrize('x, d, expected', [
    (19.38, 2, '19{,}38'),
    (100.0, 2, '100{,}00'),
    (0.5, 1, '0{,}5'),
    (-1.5, 2, '-1{,}50'),
])
def test_gpct(x, d, expected):
    assert gpct(x, d) == expected


def test_gpct_does_not_add_thousands_separator():
    # percentages never exceed 100, so no grouping is applied
    assert gpct(1234.5, 1) == '1234{,}5'


def test_gfloat_preserves_sign_of_small_negatives():
    assert gfloat(-0.5, 4) == '-0{,}5000'


def test_gfloat_keeps_sign_for_negatives_below_minus_one():
    assert gfloat(-1.5, 4) == '-1{,}5000'
