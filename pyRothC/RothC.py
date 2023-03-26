"""Python version of The Rothamsted carbon model (RothC) 26.3.
    RothC is a model for the turnover of organic carbon in non-waterlogged topsoil that allows 
    for the effects of soil type, temperature, soil moisture and plant cover on the turnover process.

    Author: Misha Grol - grol81@mail.ru
    """
from typing import Union, Tuple
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.interpolate import interp1d


class RothC:
    """Class for Python version of The Rothamsted carbon model (RothC) 26.3."""

    def __init__(
        self,
        temperature: Union[list, np.ndarray],
        precip: Union[list, np.ndarray],
        evaporation: Union[list, np.ndarray],
        years: int = 500,
        ks: np.ndarray = np.array([10, 0.3, 0.66, 0.02, 0]),
        C0: np.ndarray = np.array([0, 0, 0, 0, 2.7]),
        input_carbon: Union[float, np.ndarray] = 1.7,
        farmyard_manure: Union[float, np.ndarray] = 0,
        clay: float = 23.4,
        soil_thickness: float = 25.0,
        DR: float = 1.44,  # ratio DPM/RPM
        pE: float = 0.75,
        bare: bool = False,
        solver: str = "euler",
    ):
        """Python version of The Rothamsted carbon model (RothC) 26.3.

        Args:
            temperature (Union[list, np.ndarray]): Values of monthly temperature
                                                for which the effects on decomposition rates are calculated (°C).
            precip (Union[list, np.ndarray]): Values of monthly precipitaion
                                                for which the effects on decomposition rates are calculated (mm).
            evaporation (Union[list, np.ndarray]): Values of monthly open pan evaporation or evapotranspiration (mm).
            years (int, optional): Number of years to run RothC model. Defaults to 500.
            ks (np.ndarray, optional): A vector of length 5 containing the values of the
                                    decomposition rates for the different pools.
                                    Defaults to np.array([10, 0.3, 0.66, 0.02, 0]).
            C0 (np.ndarray, optional): A numpy of length 5 containing the initial amount
                                        of carbon for the 5 pools. Defaults to np.array([0, 0, 0, 0, 2.7]).
            input_carbon (Union[float, np.ndarray], optional): A scalar or np.array
                                                                the amount of litter inputs by time. Defaults to 1.7.
            farmyard_manure (Union[float, np.ndarray], optional): A scalar or np.array object specifying the amount
                                                                 of Farm Yard Manure inputs by time. Defaults to 0.
            clay (float, optional): Percent clay in mineral soil. Defaults to 23.4.
            soil_thickness (float, optional): Soil thickness im cm. Defaults to 25.0.
            DR (float, optional): A scalar representing the ratio of decomposable plant material
                                to resistant plant material (DPM/RPM). Defaults to 1.44.
            pE (float, optional): Evaporation coefficient.
                                If open pan evaporation is used pE=0.75.
                                If Potential evaporation is used, pE=1.0.
            bare (bool, optional): Logical. Under bare soil conditions, bare=True.
                                Default is set under vegetated soil. Defaults to False.
            solver (str, optional): Solver - Not implemented yet. Defaults to "euler".

        Raises:
            ValueError: _description_
            ValueError: _description_
        """
        self.years = years
        self.t = np.arange(1 / 12, years, step=1 / 12)
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
        self.pE = pE
        self.bare = bare
        self.soil_thickness = soil_thickness
        self.solver = solver
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

    def _get_stress_parameters(
        self, temperature: np.ndarray, precip: np.ndarray, evaporation: np.ndarray
    ) -> np.ndarray:
        """Compute decomposition impact of Temperature and Moisture (fT * fW)

        Args:
            temperature (np.ndarray): Values of monthly temperature
                                    for which the effects on decomposition
                                    rates are calculated (°C).
            precip (np.ndarray): Values of monthly precipitaion
                                for which the effects on
                                decomposition rates are calculated (mm).
            evaporation (np.ndarray): Values of monthly open pan evaporation or evapotranspiration (mm).

        Returns:
            np.ndarray: Effects of moisture and temperature
                        on decomposition rates according to the RothC model.
        """

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

    def fT(self, temperature: np.ndarray) -> list:
        """Compute Temperature factor

        Args:
            temperature (np.ndarray): monthly mean temperature

        Returns:
            list: monthly temperature factor
        """
        stress_coef = []
        for x in temperature:
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
        bare: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Soil moisture factor

        Args:
            precip (np.ndarray): Values of monthly precipitaion
                                for which the effects on 
                                decomposition rates are calculated (mm).
            evaporation (np.ndarray): Values of monthly open pan evaporation or evapotranspiration (mm).
            soil_thickness (float): Soil thickness in cm. Default for Rothamsted is 23 cm.
            pE (float, optional): Evaporation coefficient.
                                If open pan evaporation is used pE=0.75.
                                If Potential evaporation is used, pE=1.0.. Defaults to 0.75.
            clay (float, optional): Percent clay in mineral soil. Defaults to 23.4.
                                 Defaults to 20.0.
            bare (bool, optional): Logical. Under bare soil conditions, bare=True.
                                Default is set under vegetated soil.
                                Defaults to False.

        Raises:
            ValueError: Precip and evaporation have different shape

        Returns:
            Tuple[np.ndarray, np.ndarray]: _description_
        """
        # compute maximum soil moisture deficit
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
        self, input_carbon: float, farmyard_manure: float = 0, DR: float = 1.44
    ) -> np.ndarray:
        """Get amount of litter inputs

        Args:
            input_carbon (float): A scalar or np.array
                                the amount of litter inputs by time. Defaults to 1.7.
            farmyard_manure (float, optional): A scalar or np.array object specifying the amount
                                                of Farm Yard Manure inputs by time.
                                                Defaults to 0.
            DR (float, optional): A scalar representing the ratio of decomposable plant material
                                to resistant plant material (DPM/RPM). Defaults to 1.44.

        Returns:
            np.ndarray: input_DPM, input_RPM, input_BIO, input_HUM, input_IOM
        """
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
        x = 1.67 * (1.85 + 1.60 * np.exp(-0.0786 * self.clay))
        B = 0.46 / (x + 1)
        H = 0.54 / (x + 1)
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
