"""
    Python version of The Rothamsted carbon model (RothC) 26.3.
    RothC is a model for the turnover of organic carbon in non-waterlogged topsoil that allows 
    for the effects of soil type, temperature, soil moisture and plant cover on the turnover process.

    Author: Misha Grol - grol81@mail.ru
    Modifications: Omer Tzuk - We-Agri Ltd
"""
import numpy as np

class Pedotransfer:
    def __init__(self, t=1.0,theta_R=0.01) -> None:
        self.theta_R: float = theta_R
        self._t: float = t

    def t_to_percent(self, SOC_t_ha: float, BD: float, depth_cm: float):
        """
        Convert t/ha of SOC into SOC(%)

        OC(%) * 100 = Soil OC (ton/ha) /  BD (g/cm3) / Depth (cm)
        """
        SOC_percent = SOC_t_ha / BD / depth_cm
        return SOC_percent

    def calc_mbar(self, x):
        return 1000.0 * x

    def equation_alpha(self, Silt, Clay, OC, BD, t):
        return np.exp(
            - 14.96
            + 0.03135 * Clay
            + 0.0351 * Silt
            + 0.646 * (OC * 1.72)
            + 15.29 * BD
            - 0.192 * t
            - 4.671 * BD**2
            - 0.000781 * Clay**2
            - 0.00687 * (OC * 1.72)**2
            + 0.0449 * (OC * 1.72)**-1
            + 0.0663 * np.log(Silt)
            + 0.1482 * np.log(OC * 1.72)
            - 0.04546 * BD * Silt
            - 0.4852 * BD * (OC * 1.72)
            + 0.00673 * Clay * t
        )

    def equation_theta_s(self, Silt, Clay, OC, BD, t):
        return (
            0.7919
            + 0.001691 * Clay
            - 0.29619 * BD
            - 0.000001491 * Silt**2
            + 0.0000821 * (OC * 1.72)**2
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
                - 25.23
                - 0.02195 * Clay
                + 0.0074 * Silt
                - 0.194 * (OC * 1.72)
                + 45.5 * BD
                - 7.24 * BD**2
                + 0.0003658 * Clay**2
                + 0.002885 * (OC * 1.72)**2
                - 12.81 * BD**-1
                - 0.1524 * Silt**-1
                - 0.01958 * (OC * 1.72)**-1
                - 0.2876 * np.log(Silt)
                - 0.0709 * np.log(OC * 1.72)
                - 44.6 * np.log(BD)
                - 0.02264 * BD * Clay
                + 0.0896 * BD * (OC * 1.72)
                + 0.00718 * Clay * t
            )
            + 1.0
        )

    def equation_wc(self, theta_R, theta_s, alpha, n, mbar):
        print("theta_R={}, theta_s={}, alpha={}, n={}, mbar={}".format(theta_R, theta_s, alpha, n, mbar))
        print("(alpha * mbar) ** n=",(alpha * mbar) ** n)
        return theta_R + (theta_s - theta_R) #/ (
        #    (1.0 + (alpha * mbar) ** n) ** (1.0 - (1.0 / n))
        #)

    def direct_VanGen(self, theta_R, theta_s, alpha, n, mbar):
        print("theta_R={}, theta_s={}, alpha={}, n={}, mbar={}".format(theta_R, theta_s, alpha, n, mbar))
        print("(alpha * mbar) ** n=",(alpha * mbar) ** n)
        return theta_R + (theta_s - theta_R) / ((1 + (alpha * mbar) ** n)) ** (1 - 1 / n)

    def calc_WCi(self, Silt, Clay, OC, BD, theta_R, t, bar):
        theta_s = self.equation_theta_s(Silt, Clay, OC, BD, t)
        alpha = self.equation_alpha(Silt, Clay, OC, BD, t)
        n = self.equation_n(Silt, Clay, OC, BD, t)
        mbar = self.calc_mbar(bar)
        return self.direct_VanGen(theta_R, theta_s, alpha, n, mbar)

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
        OC = self.t_to_percent(SOC_t_ha = OC, BD=BD, depth_cm =soil_thickness )
        print(f'SOC (%) - {OC}')
        # Water content at field capacity
        WCfc = self.calc_WCi(Silt, Clay, OC, BD, theta_R, t, 0.05) # Q: Why mbar was negative?
        WCb = self.calc_WCi(Silt, Clay, OC, BD, theta_R, t, 1) # Q: Why mbar was negative?
        # pwp - permanent wilting point
        WCpwp = self.calc_WCi(Silt, Clay, OC, BD, theta_R, t, 15) # Q: Why mbar was negative?
        # capillary water retained at − 1000 bar
        WCc = self.calc_WCi(Silt, Clay, OC, BD, theta_R, t, 1000) # Q: Why mbar was negative?
        Mb = self.calc_M_i(WCb, WCfc, soil_thickness)
        Mpwp = self.calc_M_i(WCpwp, WCfc, soil_thickness)
        Mc = self.calc_M_i(WCc, WCfc, soil_thickness)
        return Mb, Mpwp, Mc