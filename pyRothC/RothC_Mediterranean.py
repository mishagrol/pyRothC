"""
    Python version of The Rothamsted carbon model (RothC) 26.3.
    RothC is a model for the turnover of organic carbon in non-waterlogged topsoil that allows 
    for the effects of soil type, temperature, soil moisture and plant cover on the turnover process.

    Author: Misha Grol - grol81@mail.ru
"""

import logging
from typing import Tuple, Union

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.interpolate import interp1d


"""
    Python version of The Rothamsted carbon model (RothC) 26.3.
    RothC is a model for the turnover of organic carbon in non-waterlogged topsoil that allows 
    for the effects of soil type, temperature, soil moisture and plant cover on the turnover process.

    Annex A Pedotransfer functions used to calculate the hydraulic properties from
    Farina et at 2013 - Modification of the RothC model for simulations of soil organic C dynamics in dryland regions
    https://ars.els-cdn.com/content/image/1-s2.0-S0016706113000438-mmc1.pdf

    Author: Misha Grol - grol81@mail.ru
    Modifications: Omer Tzuk 
    M_field_capacity(0.05 bar)
    M_b(1 bar)
    M(15 bar) - wilting point!
    M_c(1000 bar)
"""
import numpy as np


class Pedotransfer:
    def __init__(self) -> None:
        pass
        # self.theta_R: float = 0.01
        # self._t: float = 1.0

    def calc_mbar(self, x):
        return 1000.0 * x

    def equation_alpha(self, Silt, Clay, OC, BD, t):
        return np.exp(
            -14.96
            + 0.03135 * Silt
            + 0.0351 * Clay
            + 0.646 * OC * (BD * 1.72)
            + 15.29 * OC
            - 0.192 * BD
            - 4.671 * BD**2
            - 0.000781 * Clay**2
            - 0.00687 * (OC * 1.72) ** 2
            + 0.0449 * (OC * 1.72) ** -1
            + 0.0663 * np.log(BD)
            + 0.1482 * np.log(OC * 1.72)
            - 0.04546 * OC * BD
            - 0.4852 * OC * (BD * 1.72)
            + 0.00673 * BD * t
        )

    def equation_theta_s(self, Silt, Clay, OC, BD, t):
        return (
            0.7919
            + 0.001691 * Clay
            - 0.29619 * BD
            - 0.000001491 * Silt**2
            + 0.0000821 * (OC * 1.72) ** 2
            + 0.02427 * Clay**-1
            + 0.01113 * Silt**-1
            + 0.01472 * np.log(Silt)
            - 0.0000733 * (OC * 1.72) * Clay
            - 0.000619 * BD * Clay
            - 0.001183 * BD * (OC * 1.72)
            - 0.0001664 * Silt * t
        )

    def equation_n(self, Silt, Clay, OC, BD, t):
        return (
            np.exp(
                -25.23
                - 0.02195 * Clay
                + 0.0074 * Silt
                - 0.194 * (OC * 1.72)
                + 45.5 * OC
                - 7.24 * BD**2
                + 0.0003658 * Clay**2
                + 0.002885 * (OC * 1.72) ** 2
                - 12.81 * BD**-1
                - 0.1524 * Silt**-1
                - 0.01958 * (OC * 1.72) ** -1
                - 0.2876 * np.log(Silt)
                - 0.0709 * np.log(OC * 1.72)
                - 44.6 * np.log(BD)
                - 0.02264 * BD * Clay
                + 0.0896 * BD * (OC * 1.72)
                + 0.00718 * Clay * t
            )
            + 1
        )

    def equation_wc(self, theta_R, theta_s, alpha, n, mbar):
        return theta_R + (theta_s - theta_R) / (
            (1 + (alpha + mbar) ** n) ** (1 - (1 / n))
        )

    def calc_WCi(self, Silt, Clay, OC, BD, theta_R, t, bar):
        theta_s = self.equation_theta_s(Silt, Clay, OC, BD, t)
        alpha = self.equation_alpha(Silt, Clay, OC, BD, t)
        n = self.equation_n(Silt, Clay, OC, BD, t)
        mbar = self.calc_mbar(bar)
        return self.equation_wc(theta_R, theta_s, alpha, n, mbar)

    def calc_M_i(self, WC_i, WC_fc, depth):
        # To convert from water content to soil moisture deficit (mm) used by RothC the following equation is used
        return (WC_i - WC_fc) * 10 * depth

    def calc_Mis(
        self,
        silt,
        clay,
        BD,
        C,
        t: float = 1.0,
        theta_R: float = 0.01,
        soil_thickness: float = 23,
    ):
        # From Annex A Pedotransfer functions
        # used to calculate the hydraulic properties
        # At Farina et al 2013
        Silt = silt
        Clay = clay
        OC = np.sum(C[:-1])
        # Water content at field capacity
        WCfc = self.calc_WCi(Silt, Clay, OC, BD, theta_R, t, -0.05)
        WCb = self.calc_WCi(Silt, Clay, OC, BD, theta_R, t, -1)
        # pwp - permanent wilting point
        WCpwp = self.calc_WCi(Silt, Clay, OC, BD, theta_R, t, -15)
        # capillary water retained at − 1000 bar
        WCc = self.calc_WCi(Silt, Clay, OC, BD, theta_R, t, -1000)
        Mb = self.calc_M_i(WCb, WCfc, soil_thickness)
        Mpwp = self.calc_M_i(WCpwp, WCfc, soil_thickness)
        Mc = self.calc_M_i(WCc, WCfc, soil_thickness)
        return Mb, Mpwp, Mc


class RothC_MM(Pedotransfer):
    KS_LENGTH = 5
    C0_LENGTH = 5
    YEARS = 500
    KS_DEFAULT = np.array([10, 0.3, 0.66, 0.02, 0])
    C0_DEFAULT = np.array([0, 0, 0, 0, 2.7])
    INPUT_CARBON_DEFAULT = 1.7
    FARMYARD_MANURE_DEFAULT = 0
    CLAY_DEFAULT = 23.4
    SILT_DEFAULT = 30.0
    BD = 1.3  # Bulk density default value in g/cm^3
    SOIL_THICKNESS_DEFAULT = 25.0
    DR_DEFAULT = 1.44
    PE_DEFAULT = 0.75
    BARE_DEFAULT = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0])
    STRESS_COEF_CONSTANTS = (47.9, 106, 18.3)
    MAX_SMD_CONSTANTS = (20.0, 1.3, 0.01, 23)
    B_VALUE_CONSTANTS = (0.2, 0.8, 0.444)
    DC_DT_CONSTANTS = (1.67, 1.85, 1.60, 0.0786, 0.46, 0.54)
    MONTHS_IN_YEAR = 12

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
        silt: float = SILT_DEFAULT,
        BD: float = BD,
        soil_thickness: float = SOIL_THICKNESS_DEFAULT,
        DR: float = DR_DEFAULT,
        pE: float = PE_DEFAULT,
        bare: np.ndarray = BARE_DEFAULT,
        use_capilarity: bool = False,
        log_level: str = "ERROR",
    ):
        """
            Python version of The Rothamsted carbon model (RothC) 26.3.

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
            bare (np.ndarray, optional): Array. Under bare soil conditions, bare is 1, else is 0.
                                Default is set [0,0,0,1,1,1,1,1,1,0,0,0], means bare for warm period.
            solver (str, optional): Solver - Not implemented yet. Defaults to "euler".

        Raises:
            ValueError: _description_
            ValueError: _description_
        """

        logging.basicConfig(
            format=(
                "%(asctime)s, %(levelname)-8s"
                "[%(filename)s:%(module)s:%(funcName)s"
                ":%(lineno)d] %(message)s"
            ),
            datefmt="%Y-%m-%d:%H:%M:%S",
        )
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
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
        self.silt = silt
        self.BD = BD
        self.sand = 100.0 - (clay + silt)
        self.DR = DR
        self.pE = pE
        self.bare = bare
        self.use_capilarity = use_capilarity
        self.soil_thickness = soil_thickness
        self._t: list = []
        self.xi = self._get_stress_parameters(
            temperature=np.array(temperature),
            precip=np.array(precip),
            evaporation=np.array(evaporation),
        )

        self.xi_func = interp1d(
            self.t, self.xi, fill_value="extrapolate"  # type: ignore
        )
        self._current_XI: list = []

    @staticmethod
    def validate_ks(ks: np.ndarray):
        if len(ks) != RothC_MM.KS_LENGTH:
            raise ValueError("ks must be of length = 5")

    @staticmethod
    def validate_C0(C0: np.ndarray):
        if len(C0) != RothC_MM.C0_LENGTH:
            raise ValueError("the vector with initial conditions must be of length = 5")

    def _get_stress_parameters(
        self, temperature: np.ndarray, precip: np.ndarray, evaporation: np.ndarray
    ) -> np.ndarray:
        """
            Compute decomposition impact of Temperature and Moisture (fT * fW)

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
        acc_TSMD, b = self.van_Genuchten_fW(
            precip=precip,
            evaporation=evaporation,
            soil_thickness=self.soil_thickness,
        )
        self._stress_temp = stress_temp
        self._b = b
        xi = stress_temp * b
        xi = np.tile(xi, self.years)
        return xi

    def fT(self, temperature: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            stress_coef = self.STRESS_COEF_CONSTANTS[0] / (
                1
                + np.exp(
                    self.STRESS_COEF_CONSTANTS[1]
                    / (temperature + self.STRESS_COEF_CONSTANTS[2])
                )
            )
            stress_coef[temperature < -self.STRESS_COEF_CONSTANTS[2]] = np.nan
        return stress_coef

    def van_Genuchten_fW(
        self,
        precip: np.ndarray,
        evaporation: np.ndarray,
        soil_thickness: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
            Compute Soil moisture factor with Van Genuchten

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
            bare (np.ndarray, optional): Array. Under bare soil conditions, bare is 1, else is 0.
                                Default is set [0,0,0,1,1,1,1,1,1,0,0,0], means bare for summer month.


        Raises:
            ValueError: Precip and evaporation have different shape

        Returns:
            Tuple[np.ndarray, np.ndarray]: _description_
        """

        # compute maximum soil moisture deficit
        if precip.shape != evaporation.shape:
            raise ValueError("Precip and evaporation arrays must have the same shape")

        Mb, M, Mc = self.calc_Mis(
            silt=self.silt,
            clay=self.clay,
            BD=self.BD,
            C=self.C0,
            t=1.0,
            theta_R=0.01,
            soil_thickness=self.soil_thickness,
        )

        if self.use_capilarity:
            M = Mc
        # if self.bare:
        #     M /= 1.8

        M = np.full(self.MONTHS_IN_YEAR, M)
        mask = self.bare.astype(bool)
        M[mask] /= 1.8
        acc_TSMD = np.zeros_like(M, dtype=np.float64)
        acc_TSMD[0] = min(M[0], 0)
        for i in range(1, len(M)):
            acc_TSMD[i] = min(acc_TSMD[i - 1] + M[i], 0)
            acc_TSMD[i] = max(acc_TSMD[i], M[i])
            # acc_TSMD[i] = max(acc_TSMD[i], max_smd)

        # from paper : b  =  0.2 + 0.8 * (M - AccM ) / (M - Mb)
        b = self.B_VALUE_CONSTANTS[0] + self.B_VALUE_CONSTANTS[1] * (M - acc_TSMD) / (
            M - Mb
        )
        b[acc_TSMD > self.B_VALUE_CONSTANTS[2] * M] = 1

        return acc_TSMD, b

    def fW(
        self,
        precip: np.ndarray,
        evaporation: np.ndarray,
        soil_thickness: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
            Compute Soil moisture factor

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
            bare (np.ndarray, optional): Array. Under bare soil conditions, bare is 1, else is 0.
                                Default is set [0,0,0,1,1,1,1,1,1,0,0,0], means bare for summer month.


        Raises:
            ValueError: Precip and evaporation have different shape

        Returns:
            Tuple[np.ndarray, np.ndarray]: _description_
        """

        # compute maximum soil moisture deficit
        if precip.shape != evaporation.shape:
            raise ValueError("Precip and evaporation arrays must have the same shape")
        # From paper: -(20 + 1.3 * Clay - 0.01 * clay **2) * depth / 23
        max_smd_value = -(
            self.MAX_SMD_CONSTANTS[0]
            + self.MAX_SMD_CONSTANTS[1] * self.clay
            - self.MAX_SMD_CONSTANTS[2] * self.clay**2
        ) * (soil_thickness / self.MAX_SMD_CONSTANTS[3])
        # From paper: Mbare = M / 1.8

        max_smd = np.full(self.MONTHS_IN_YEAR, max_smd_value)
        mask = self.bare.astype(bool)
        max_smd[mask] /= 1.8
        # max_smd = max_smd_value
        # if self.bare:
        #     max_smd /= 1.8

        M = precip - evaporation * self.pE
        acc_TSMD = np.zeros_like(M, dtype=np.float64)
        acc_TSMD[0] = min(M[0], 0)
        for i in range(1, len(M)):
            acc_TSMD[i] = min(acc_TSMD[i - 1] + M[i], 0)
            acc_TSMD[i] = max(acc_TSMD[i], max_smd[i])
            # acc_TSMD[i] = max(acc_TSMD[i], max_smd)

        # from paper : b  =  0.2 + 0.8 * (M - AccM ) / (M - Mb)
        b = self.B_VALUE_CONSTANTS[0] + self.B_VALUE_CONSTANTS[1] * (
            max_smd - acc_TSMD
        ) / (max_smd - self.B_VALUE_CONSTANTS[2] * max_smd)
        b[acc_TSMD > self.B_VALUE_CONSTANTS[2] * max_smd] = 1

        return acc_TSMD, b

    def get_input_flux(
        self,
        input_carbon: Union[float, np.ndarray],
        farmyard_manure: Union[float, np.ndarray] = 0,
        DR: Union[float, np.ndarray] = 1.44,
    ) -> np.ndarray:
        """
            Get amount of litter inputs

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
        input_DPM = np.sum(input_DPM)
        input_RPM = input_carbon * (1 - gamma) + (farmyard_manure * 0.49)
        input_RPM = np.sum(input_RPM)
        input_BIO = 0
        input_HUM = farmyard_manure * 0.02
        input_HUM = np.sum(input_HUM)
        input_IOM = 0
        return np.array(
            [
                input_DPM,
                input_RPM,
                input_BIO,
                input_HUM,
                input_IOM,
            ]
        )

    def dCdt(
        self,
        C,
        t,
        input_carbon: Union[float, np.ndarray],
        farmyard_manure: Union[float, np.ndarray],
        DR: Union[float, np.ndarray],
    ):
        """
        Calculates the rate of change of soil carbon over time.

        This function is invoked by the ODE integration function to compute changes in carbon
        concentrations across various pools in the RothC model over time.

        Args:
            C (np.ndarray): An array representing the current concentrations of carbon in different pools.
            t (float): The current time in the simulation.
            input_carbon (Union[float, np.ndarray]): The amount of incoming carbon, either as a fixed value or a time-varying array.
            farmyard_manure (Union[float, np.ndarray]): The amount of farmyard manure applied, again either as a fixed value or a time-varying array.
            DR (Union[float, np.ndarray]): The decomposition rate, representing the ratio of decomposable to resistant plant material.

        Returns:
            np.ndarray: An array representing the rate of change of carbon in each pool at time `t`.

        This function uses constants defined in the class to calculate the decomposition coefficients (B and H) and
        then applies these coefficients along with the decomposition rates (`ks`) to compute changes in carbon
        concentrations. It also calculates the incoming carbon fluxes and adjusts them based on environmental stress
        factors calculated by `xi_func`.
        """
        self._t.append(t)
        ks = self.ks
        x = self.DC_DT_CONSTANTS[0] * (
            self.DC_DT_CONSTANTS[1]
            + self.DC_DT_CONSTANTS[2] * np.exp(-self.DC_DT_CONSTANTS[3] * self.clay)
        )
        B = self.DC_DT_CONSTANTS[4] / (x + 1)
        H = self.DC_DT_CONSTANTS[5] / (x + 1)
        ai3 = B * ks
        ai4 = H * ks
        A = np.diag(-ks)
        A[2] = A[2] + ai3
        A[3] = A[3] + ai4
        in_flux = self.get_input_flux(input_carbon, farmyard_manure, DR)
        xi_f = self.xi_func(t)
        self._current_XI.append(xi_f)
        C_next = in_flux + (A * xi_f).dot(C)
        return C_next

    def compute(self):
        y1 = odeint(
            self.dCdt,
            self.C0,
            t=self.t,
            args=(self.input_carbon, self.farmyard_manure, self.DR),
            rtol=0.01,
            atol=0.01,
        )
        return pd.DataFrame(y1, columns=self.ks_pulls)
