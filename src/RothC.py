from typing import Union
import numpy as np

from typing import Union
import numpy as np


class RothC:
    def __init__(
        self,
        years: int = 500,
        ks: np.ndarray = np.array([10, 0.3, 0.66, 0.02, 0]),
        C0: np.ndarray = np.array([0, 0, 0, 0, 2.7]),
        input_carbon: Union[float, np.ndarray] = 1.7,
        farmyard_manure: Union[float, np.ndarray] = 0,
        clay: float = 23.4,
        DR: float = 1.44,  # ratio DPM/RPM
        xi: float = 1.0,
        solver: str = "euler",
    ):
        self.years = np.arange(1 / 12, years, step=1 / 12)
        self.t = self.years
        self.ks_pulls = ["DPM", "RPM", "BIO", "HUM", "IOM"]
        if len(ks) != 5:
            raise ValueError("ks must be of length = 5")
        self.ks = ks
        if len(C0) != 5:
            raise ValueError("the vector with initial conditions must be of length = 5")
        self.C0 = C0
        self.farmyard_manure = farmyard_manure
        self.input_carbon = input_carbon
        self.clay = clay
        self.DR = DR
        self.xi = xi  # stress factor by Soil Moisture * Temperature
        self.solver = solver
        t_start = min(self.t)
        t_end = max(self.t)

    def fT(self, temperture: list) -> list:
        stress_coef = []
        for x in temperture:
            if x < -18.3:
                stress_coef.append(np.nan)
            else:
                stress_coef.append(47.9 / (1 + np.exp(106 / (x + 18.3))))
        return stress_coef

    def fW(
        self,
        precip: np.ndarray,
        evaporation: np.ndarray,
        soil_thickness: float,
        pE: float = 0.75,
        clay: float = 20.0,
        bare: bool = True,
    ):
        """
        P - A vector with monthly precipitation (mm).

        E A vector with same length with open pan evaporation or evapotranspiration (mm).

        S.Thick
        Soil thickness in cm. Default for Rothamsted is 23 cm.

        pClay
        Percent clay.

        pE
        Evaporation coefficient. If open pan evaporation is used pE=0.75. If Potential evaporation is used, pE=1.0.

        bare
        Logical. Under bare soil conditions, bare=TRUE. Default is set under vegetated soil.

        """
        # compute maximum soil moisture defficit
        if precip.shape != evaporation.shape:
            raise ValueError("Precip and evaporation have different shape")
        #     max_smd = -(20.0 + 1.3 * clay - 0.01 * clay**2)
        max_smd = -(20.0 + 1.3 * clay - 0.01 * clay**2) * (
            soil_thickness / 23
        )  # TO-DO Verify this soil_thik / 23
        if bare:
            max_smd = max_smd / 1.8
        M = precip - evaporation * pE
        acc_TSMD = np.zeros(shape=(len(M)))
        if M[0] > 0:
            acc_TSMD[0] = 0
        else:
            acc_TSMD[0] = M[0]
        for i in range(1, len(M)):
            if acc_TSMD[i - 1] + M[i] < 0:
                acc_TSMD[i] = acc_TSMD[i - 1] + M[i]
            else:
                acc_TSMD[i] = 0
            if acc_TSMD[i] <= max_smd:
                acc_TSMD[i] = max_smd
        b = np.zeros(shape=acc_TSMD.shape)
        mask = acc_TSMD > 0.444 * max_smd
        b = 0.2 + 0.8 * ((max_smd - acc_TSMD) / (max_smd - 0.444 * max_smd))
        b[mask] = 1
        return acc_TSMD, b

    def get_input_flux(
        self, input_carbon: float, farmyard_manure: float, DR: float
    ) -> dict:
        """
        input_carbon (float): The plant residue input is the amount of carbon that is put into the soil per month
        farmyard_manure (float): monthly input of farmyard manure (FYM) (t C ha-1),
        DR (float): # ratio DPM/RPM
        """
        gamma = DR / (1 + DR)
        input_DPM = input_carbon * gamma + (farmyard_manure * 0.49)
        input_RPM = input_carbon * (1 - gamma) + (farmyard_manure * 0.49)
        input_BIO = 0
        input_HUM = farmyard_manure * 0.02
        input_IOM = 0
        return np.array([input_DPM, input_RPM, input_BIO, input_HUM, input_IOM])
