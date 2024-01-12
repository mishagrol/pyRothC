from typing import Union, Tuple
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.interpolate import interp1d


class RothC:

    KS_LENGTH = 5
    C0_LENGTH = 5
    YEARS = 500
    KS_DEFAULT = np.array([10, 0.3, 0.66, 0.02, 0])
    C0_DEFAULT = np.array([0, 0, 0, 0, 2.7])
    INPUT_CARBON_DEFAULT = 1.7
    FARMYARD_MANURE_DEFAULT = 0
    CLAY_DEFAULT = 23.4
    SOIL_THICKNESS_DEFAULT = 25.0
    DR_DEFAULT = 1.44
    PE_DEFAULT = 0.75
    BARE_DEFAULT = False
    STRESS_COEF_CONSTANTS = (47.9, 106, 18.3)
    MAX_SMD_CONSTANTS = (20.0, 1.3, 0.01, 23)
    B_VALUE_CONSTANTS = (0.2, 0.8, 0.444)
    DC_DT_CONSTANTS = (1.67, 1.85, 1.60, 0.0786, 0.46, 0.54)

    def __init__(
        self,
        temperature: Union[list, np.ndarray],
        precip: Union[list, np.ndarray],
        evaporation: Union[list, np.ndarray],
        years: int = YEARS,
        ks: np.ndarray = KS_DEFAULT,
        C0: np.ndarray = C0_DEFAULT,
        input_carbon: Union[float, np.ndarray] = INPUT_CARBON_DEFAULT,
        farmyard_manure: Union[float, np.ndarray] = FARMYARD_MANURE_DEFAULT,
        clay: float = CLAY_DEFAULT,
        soil_thickness: float = SOIL_THICKNESS_DEFAULT,
        DR: float = DR_DEFAULT,
        pE: float = PE_DEFAULT,
        bare: bool = BARE_DEFAULT,
    ):

        self.validate_ks(ks)
        self.validate_C0(C0)

        self.years = years
        self.t = np.linspace(1 / 12, years, num=years * 12)
        self.ks_pulls = ["DPM", "RPM", "BIO", "HUM", "IOM"]
        self.ks = ks
        self.C0 = C0
        self.farmyard_manure = farmyard_manure
        self.input_carbon = input_carbon
        self.clay = clay
        self.DR = DR
        self.pE = pE
        self.bare = bare
        self.soil_thickness = soil_thickness
        self._t = []
        self.xi = self._get_stress_parameters(
            temperature=np.array(temperature),
            precip=np.array(precip),
            evaporation=np.array(evaporation),
        )

        self.xi_func = interp1d(
            self.t, self.xi, fill_value="extrapolate"  # type: ignore
        )
        self._current_XI = []
    
        @staticmethod
        def validate_ks(ks: np.ndarray):
            if len(ks) != RothC.KS_LENGTH:
                raise ValueError("ks must be of length = 5")

        @staticmethod
        def validate_C0(C0: np.ndarray):
            if len(C0) != RothC.C0_LENGTH:
                raise ValueError("the vector with initial conditions must be of length = 5")

    def _get_stress_parameters(
        self, temperature: np.ndarray, precip: np.ndarray, evaporation: np.ndarray
    ) -> np.ndarray:

        stress_temp = self.fT(temperature=temperature)
        acc_TSMD, b = self.fW(
            precip=precip,
            evaporation=evaporation,
            pE=self.pE,
            bare=self.bare,
            clay=self.clay,
            soil_thickness=self.soil_thickness,
        )
        self._stress_temp = stress_temp
        self._b = b
        xi = stress_temp * b
        xi = np.tile(xi, self.years)
        return xi

    def fT(self, temperature: np.ndarray) -> np.ndarray:
        with np.errstate(divide='ignore', invalid='ignore'):
            stress_coef = self.STRESS_COEF_CONSTANTS[0] / (1 + np.exp(self.STRESS_COEF_CONSTANTS[1] / (temperature + self.STRESS_COEF_CONSTANTS[2])))
            stress_coef[temperature < -self.STRESS_COEF_CONSTANTS[2]] = np.nan
        return stress_coef

    def fW(
        self,
        precip: np.ndarray,
        evaporation: np.ndarray,
        soil_thickness: float,
        pE: float = 0.75,
        clay: float = 20.0,
        bare: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:

        # compute maximum soil moisture deficit
        if precip.shape != evaporation.shape:
            raise ValueError("Precip and evaporation arrays must have the same shape")

        max_smd = -(self.MAX_SMD_CONSTANTS[0] + self.MAX_SMD_CONSTANTS[1] * self.clay - self.MAX_SMD_CONSTANTS[2] * self.clay**2) * (soil_thickness / self.MAX_SMD_CONSTANTS[3])
        if self.bare:
            max_smd /= 1.8

        M = precip - evaporation * self.pE
        acc_TSMD = np.minimum.accumulate(M.clip(max=0))
        acc_TSMD = np.maximum(acc_TSMD, max_smd)

        b = self.B_VALUE_CONSTANTS[0] + self.B_VALUE_CONSTANTS[1] * (max_smd - acc_TSMD) / (max_smd - self.B_VALUE_CONSTANTS[2] * max_smd)
        b[acc_TSMD > self.B_VALUE_CONSTANTS[2] * max_smd] = 1

        return acc_TSMD, b

    def get_input_flux(
        self, input_carbon: float, farmyard_manure: float = 0, DR: float = 1.44
    ) -> np.ndarray:

        gamma = DR / (1 + DR)
        input_DPM = input_carbon * gamma + (farmyard_manure * 0.49)
        input_RPM = input_carbon * (1 - gamma) + (farmyard_manure * 0.49)
        input_BIO = 0
        input_HUM = farmyard_manure * 0.02
        input_IOM = 0
        return np.array([input_DPM, input_RPM, input_BIO, input_HUM, input_IOM])

    def dCdt(self, C, t):
        self._t.append(t)
        ks = self.ks
        x = self.DC_DT_CONSTANTS[0] * (self.DC_DT_CONSTANTS[1] + self.DC_DT_CONSTANTS[2] * np.exp(-self.DC_DT_CONSTANTS[3] * self.clay))
        B = self.DC_DT_CONSTANTS[4] / (x + 1)
        H = self.DC_DT_CONSTANTS[5] / (x + 1)
        ai3 = B * ks
        ai4 = H * ks
        A = np.diag(-ks)
        A[2] = A[2] + ai3
        A[3] = A[3] + ai4
        in_flux = self.get_input_flux(
            self.input_carbon, self.farmyard_manure, self.DR  # type: ignore
        )  # TO-DO: add not only scalar
        xi_f = self.xi_func(t)
        self._current_XI.append(xi_f)
        C_next = in_flux + (A * xi_f).dot(C)
        return C_next

    def compute(self):
        y1 = odeint(self.dCdt, self.C0, t=self.t, rtol=0.01, atol=0.01)
        return pd.DataFrame(y1, columns=self.ks_pulls)
