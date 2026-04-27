from ansys.mapdl.core import launch_mapdl
import numpy as np

mapdl = launch_mapdl()

height = 1
total_len = 2

class CAEModel:
    def __init__(self, paras):
        self.height = height
        self.total_len = total_len
        self.len1 = paras[0]
        self.width1 = paras[1]
        self.width2 = paras[2]
        self.points = []
        self.areas = []

    def setup(self):
        mapdl.units("SI")
        mapdl.clear()
        mapdl.prep7()

    def create_keypoints(self):
        self.points = [
            mapdl.k("", 0, 0, 0),
            mapdl.k("", self.total_len, 0, 0),
            mapdl.k("", self.total_len, self.height, 0),
            mapdl.k("", 0, self.height, 0)
        ]
        # mapdl.kplot()

    def create_areas(self):
        a1 = mapdl.a(self.points[0], self.points[1], self.points[2], self.points[3])
        self.areas.append(a1)

        # Defining keypoints and areas as per the original logic
        keypoints_and_areas = [
            (self.len1 / 2, self.height / 2, 0),
            (0, self.width1, 0, 0, self.height - self.width1, 0, self.len1 / 2 - self.width1, self.height / 2, 0),
            (self.len1 - self.width1, 2 * self.width1, 0, self.len1 - self.width1, self.height - 2 * self.width1, 0, self.len1 / 2 + self.width1, self.height / 2, 0),
            (2 * self.width1, self.width1, 0, self.len1 - 2 * self.width1, self.width1, 0, self.len1 / 2, self.height / 2 - self.width1, 0),
            (2 * self.width1, self.height - self.width1, 0, self.len1 - 2 * self.width1, self.height - self.width1, 0, self.len1 / 2, self.height / 2 + self.width1, 0),
            (self.len1, self.height - 2 * self.width2, 0, self.len1, 2 * self.width2, 0, (self.len1 + self.total_len - self.width2) / 2 - self.width2, self.height / 2, 0),
            (self.total_len - self.width2, self.height - 2 * self.width2, 0, self.total_len - self.width2, 2 * self.width2, 0, (self.len1 + self.total_len - self.width2) / 2 + self.width2, self.height / 2, 0),
            (self.len1 + self.width2, self.height - self.width2, 0, self.total_len - 2 * self.width2, self.height - self.width2, 0, (self.len1 + self.total_len - self.width2) / 2, self.height / 2 + self.width2, 0),
            (self.len1 + self.width2, self.width2, 0, self.total_len - 2 * self.width2, self.width2, 0, (self.len1 + self.total_len - self.width2) / 2, self.height / 2 - self.width2, 0)
        ]

        for i, values in enumerate(keypoints_and_areas):
            if len(values) == 3:
                p = mapdl.k("", values[0], values[1], values[2])
            else:
                t11, t12, t13 = mapdl.k("", values[0], values[1], values[2]), mapdl.k("", values[3], values[4], values[5]), mapdl.k("", values[6], values[7], values[8])
                ta = mapdl.a(t11, t12, t13)
                a_new = mapdl.asba(self.areas[-1], ta)
                self.areas.append(a_new)
        # mapdl.aplot(cpos='xy')

    def create_volume(self):
        mapdl.vext(na1=self.areas[-1], dz=self.width1)
        mapdl.mp('EX', 1, 210E9)  # Young's modulus in Pa
        mapdl.mp('NUXY', 1, 0.3)  # Poisson's ratio
        mapdl.et(1, "SOLID187")
        mapdl.esize(self.total_len / 100)
        mapdl.vmesh('ALL')
        # mapdl.vplot(cpos='XY')

    def apply_constraints_and_loads(self):
        mapdl.nsel('S', 'LOC', 'X', 0)
        mapdl.d('ALL', 'ALL')

        mapdl.nsel('S', 'LOC', 'X', self.len1)
        mapdl.f('ALL', 'FY', -1000)

        mapdl.nsel('S', 'LOC', 'X', self.total_len)
        mapdl.f('ALL', 'FY', -2000)

        mapdl.nsel('S', 'LOC', 'Y', 0)
        mapdl.f('ALL', 'FX', 1000)

    def solve(self):
        mapdl.allsel(mute=True)
        mapdl.run('/SOLU')
        mapdl.antype('STATIC')
        mapdl.solve()
        mapdl.finish()

    def post_process(self):
        result = mapdl.result
        # result.plot_principal_nodal_stress(
        #     0,
        #     "SEQV",
        #     lighting=False,
        #     background="w",
        #     show_edges=True,
        #     text_color="k",
        #     add_text=False,
        # )
        x_disp = mapdl.post_processing.nodal_displacement('X')
        max_disp_x = np.max(x_disp)
        print(f"max_disp_x = {max_disp_x}")
        return max_disp_x

    def calculate_volume(self):
        mapdl.allsel()
        volume = mapdl.get('vol', 'VOLU', 'ALL', 'VOLU')
        print(f"The volume of the geometry is {volume} cubic meters.")
        return volume

    def mapping_func(self):
        self.setup()
        self.create_keypoints()
        self.create_areas()
        self.create_volume()
        mapdl.vsum()
        self.apply_constraints_and_loads()
        self.solve()
        return self.post_process(), self.calculate_volume()

if __name__ == "__main__":
    # 使用示例
    model = CAEModel((0.7, 0.02, 0.015))
    model.mapping_func()
    mapdl.exit()