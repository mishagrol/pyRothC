"""
    Python version of The Rothamsted carbon model (RothC) 26.3.
    RothC is a model for the turnover of organic carbon in non-waterlogged topsoil that allows 
    for the effects of soil type, temperature, soil moisture and plant cover on the turnover process.

    Author: Misha Grol - grol81@mail.ru
    Modifications: Omer Tzuk - We-Agri Ltd
"""

from typing import Union, Tuple
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from pyRothC.pedotransfer_functions import Pedotransfer

class soil_texture:
    def __init__(self, clay, silt):
        self.clay = clay
        self.silt = silt
        self.sand = 100.0 - (clay+silt)    

class RothC_MM:

    KS_LENGTH = 5
    C0_LENGTH = 5
    YEARS = 500
    KS_DEFAULT = np.array([10, 0.3, 0.66, 0.02, 0])
    C0_DEFAULT = np.array([0, 0, 0, 0, 2.7])
    INPUT_CARBON_DEFAULT = 1.7
    FARMYARD_MANURE_DEFAULT = 0
    CLAY_DEFAULT = 23.4
    SILT_DEFAULT = 30.0
    BD = 1.3 # Bulk density default value in g/cm^3
    SOIL_THICKNESS_DEFAULT = 25.0
    DR_DEFAULT = 1.44
    PE_DEFAULT = 0.75
    BARE_DEFAULT = False
    STRESS_COEF_CONSTANTS = (47.9, 106, 18.3)
    MAX_SMD_CONSTANTS = (20.0, 1.3, 0.01, 23) # Maximum soil moisture deficit
    B_VALUE_CONSTANTS = (0.2, 0.8, 0.444) # Factors for the rate modifying factor b
    MIN_B_VALUE = 0.2
    DC_DT_CONSTANTS = (1.67, 1.85, 1.60, 0.0786, 0.46, 0.54)
    VERSION_DEFAULT = "RothC"

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
            bare: bool = BARE_DEFAULT,
            b_min: float = MIN_B_VALUE,
            version: str = VERSION_DEFAULT
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
                bare (bool, optional): Logical. Under bare soil conditions, bare=True.
                                    Default is set under vegetated soil. Defaults to False.
                solver (str, optional): Solver - Not implemented yet. Defaults to "euler".

            Raises:
                ValueError: _description_
                ValueError: _description_
        """
        self.validate_ks(ks)
        self.validate_C0(C0)
        self.years = years
        self.t = np.linspace(1 / 12, years, num=years * 12)
        self.ks_pulls = ["DPM", "RPM", "BIO", "HUM", "IOM"]
        self.ks = ks
        self.C0 = C0
        self.C_current = C0
        self.farmyard_manure = farmyard_manure
        self.input_carbon = input_carbon
        self.clay = clay
        self.soil_texture = soil_texture(clay, silt)
        self.BD = BD
        self.DR = DR
        self.pE = pE
        self.bare = bare
        self.b_min = b_min
        self.soil_thickness = soil_thickness
        self.version = version
        self.set_fW_version(self.version)
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


    def set_fW_version(self, version: str):
        if version not in ["RothC","RothC_Farina2013"]:
            raise ValueError("Version does not exist")
        elif version == "RothC_Farina2013":
            self.fW = self.fW_RothC_Farina2013
            self.pf = Pedotransfer(t=1.0, theta_R=0.01)
        else:
            self.fW = self.fW_RothC
    
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
        Variable names:
            accTSMD: Accumulated (Top) Soil Moisture Deficit
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
    
    def calc_Mis(self, soil_texture, C):
        # From Annex A Pedotransfer functions
        # used to calculate the hydraulic properties
        # At Farina et al 2013
        Silt = soil_texture.silt
        Clay = soil_texture.clay
        BD = self.BD
        t = theta_R.t
        theta_R = self.pf.theta_R
        OC = np.sum(C[:-1])
        depth = self.soil_thickness
        # Water content at field capacity
        WCfc = self.pf.calc_WCi(Silt, Clay, OC, BD, theta_R, t, -0.05)
        WCb  = self.pf.calc_WCi(Silt, Clay, OC, BD, theta_R, t, -1)
        # pwp - permanent wilting point
        WCpwp  = self.pf.calc_WCi(Silt, Clay, OC, BD, theta_R, t, -15)
        # capillary water retained at − 1000 bar
        WCc  = self.pf.calc_WCi(Silt, Clay, OC, BD, theta_R, t, -1000)
        Mb = self.pf.calc_M_i(WCb, WCfc, depth)
        Mpwp = self.pf.calc_M_i(WCpwp, WCfc, depth)
        Mc = self.pf.calc_M_i(WCc, WCfc, depth)
        return Mb, Mpwp, Mc

    def fT(self, temperature: np.ndarray) -> np.ndarray:
        with np.errstate(divide='ignore', invalid='ignore'):
            stress_coef = self.STRESS_COEF_CONSTANTS[0] / (1 + np.exp(self.STRESS_COEF_CONSTANTS[1] / (temperature + self.STRESS_COEF_CONSTANTS[2])))
            stress_coef[temperature < -self.STRESS_COEF_CONSTANTS[2]] = np.nan
        return stress_coef

    def fW_RothC_Farina2013(
        self,
        precip: np.ndarray,
        evaporation: np.ndarray,
        soil_thickness: float,
        pE: float = 0.75,
        clay: float = 20.0,
        bare: bool = False,
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
            raise ValueError("Precip and evaporation arrays must have the same shape")
        
        Mb, Mpwp, Mc = self.calc_Mis(self.soil_texture,self.C_current)
        # Calculating the Maximum soil moisture deficit - Eq. (1) at Farina 2013
        max_smd = -(self.MAX_SMD_CONSTANTS[0] + self.MAX_SMD_CONSTANTS[1] * self.clay - self.MAX_SMD_CONSTANTS[2] * self.clay**2) * (soil_thickness / self.MAX_SMD_CONSTANTS[3])
        # RothC takes this into account by not allowing the soil to dry out further than Mbare
        if self.bare:
            max_smd /= 1.8

        M = precip - evaporation * self.pE
        acc_TSMD = np.minimum.accumulate(M.clip(max=0))
        acc_TSMD = np.maximum(acc_TSMD, max_smd)
        # Calculating the rate modifying factor b - Eq. (3) at Farina 2013
        b = self.B_VALUE_CONSTANTS[0] + self.B_VALUE_CONSTANTS[1] * (max_smd - acc_TSMD) / (max_smd - self.B_VALUE_CONSTANTS[2] * max_smd)
        b[acc_TSMD > self.B_VALUE_CONSTANTS[2] * max_smd] = 1

        return acc_TSMD, b

    def fW_RothC(
        self,
        precip: np.ndarray,
        evaporation: np.ndarray,
        soil_thickness: float,
        pE: float = 0.75,
        clay: float = 20.0,
        bare: bool = False,
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
            raise ValueError("Precip and evaporation arrays must have the same shape")
        # Calculating the Maximum soil moisture deficit - Eq. (1) at Farina 2013
        max_smd = -(self.MAX_SMD_CONSTANTS[0] + self.MAX_SMD_CONSTANTS[1] * self.clay - self.MAX_SMD_CONSTANTS[2] * self.clay**2) * (soil_thickness / self.MAX_SMD_CONSTANTS[3])
        # RothC takes this into account by not allowing the soil to dry out further than Mbare
        if self.bare:
            max_smd /= 1.8

        M = precip - evaporation * self.pE
        acc_TSMD = np.minimum.accumulate(M.clip(max=0))
        acc_TSMD = np.maximum(acc_TSMD, max_smd)
        # Calculating the rate modifying factor b - Eq. (3) at Farina 2013
        b = self.B_VALUE_CONSTANTS[0] + self.B_VALUE_CONSTANTS[1] * (max_smd - acc_TSMD) / (max_smd - self.B_VALUE_CONSTANTS[2] * max_smd)
        b[acc_TSMD > self.B_VALUE_CONSTANTS[2] * max_smd] = 1

        return acc_TSMD, b

    def get_input_flux(
        self, input_carbon: float, farmyard_manure: float = 0, DR: float = 1.44
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
        input_RPM = input_carbon * (1 - gamma) + (farmyard_manure * 0.49)
        input_BIO = 0
        input_HUM = farmyard_manure * 0.02
        input_IOM = 0
        return np.array([input_DPM, input_RPM, input_BIO, input_HUM, input_IOM])

    def dCdt(self, C, t, input_carbon: Union[float, np.ndarray], farmyard_manure: Union[float, np.ndarray], DR: Union[float, np.ndarray]):
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
        self.C_current = C
        ks = self.ks
        x = self.DC_DT_CONSTANTS[0] * (self.DC_DT_CONSTANTS[1] + self.DC_DT_CONSTANTS[2] * np.exp(-self.DC_DT_CONSTANTS[3] * self.clay))
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
        dCdt_current = in_flux + (A * xi_f).dot(C)
        return dCdt_current

    def compute(self):
        y1 = odeint(self.dCdt, self.C_current, t=self.t, rtol=0.01, atol=0.01,
                    args=(self.input_carbon,self.farmyard_manure,self.DR))
        self.C_current = y1[-1]
        return pd.DataFrame(y1, columns=self.ks_pulls)