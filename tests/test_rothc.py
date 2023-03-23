import numpy as np
from pandas.testing import assert_frame_equal

from pyRothC.RothC import RothC


def test_compute_successful(results_2years):
    """Compare the results of a `RothC.compute` execution with the expected results."""
    temperature = np.array([-0.4, 0.3, 4.2, 8.3, 13.0, 15.9, 18.0, 17.5, 13.4, 8.7, 3.9, 0.6])
    precipitation = np.array([49, 39, 44, 41, 61, 58, 71, 58, 51, 48, 50, 58])
    evaporation = np.array([12, 18, 35, 58, 82, 90, 97, 84, 54, 31, 14, 10])
    inert_organic_matter = 0.049 * 69.7 ** 1.139

    rothc = RothC(
        temperature=temperature,
        precip=precipitation,
        evaporation=evaporation,
        years=2,  # Test the results for 2 years for simplicity.
        clay=48,
        input_carbon=2.7,
        pE=1.0,
        C0=np.array([0, 0, 0, 0, inert_organic_matter])
    )

    dataframe = rothc.compute()

    assert_frame_equal(dataframe, results_2years)
