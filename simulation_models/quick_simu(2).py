from ansys.mapdl.core import launch_mapdl
import numpy as np
mapdl = launch_mapdl()

class CAE:
    def __init__(self, lens):
        # Initialize the MAPDL object
        mapdl.clear()

        # Setup the variables
        self.len1 = lens[0]
        self.len2 = lens[1]
        self.total_len = 2
        self.width_1 = 0.4
        self.width_2 = 0.2
        self.width_3 = 0.1

    def modeling(self):
        # Define material properties
        mapdl.prep7()
        mapdl.mp('EX', 1, 210E9)  # Young's modulus in Pa
        mapdl.mp('NUXY', 1, 0.3)  # Poisson's ratio
        # Define the dimensions of each rectangular area
        self.len3 = self.total_len - self.len1 - self.len2
        r1 = mapdl.block(-self.width_1 / 2, self.width_1 / 2, -self.width_1 / 2, self.width_1 / 2, 0, self.len1)
        r2 = mapdl.block(-self.width_2 / 2, self.width_2 / 2, -self.width_2 / 2, self.width_2 / 2, self.len1,
                              self.len1 + self.len2)
        r3 = mapdl.block(-self.width_3 / 2, self.width_3 / 2, -self.width_3 / 2, self.width_3 / 2,
                              self.len1 + self.len2, self.len1 + self.len2 + self.len3)

        # Glue the volumes together and plot the geometry
        beam = mapdl.vglue('ALL')
        mapdl.et(1, "SOLID187")
        mapdl.esize(self.total_len / 50)
        mapdl.vmesh('ALL')

    def simulate(self):
        # Apply constraints (fix the face at one end of the beam)
        mapdl.nsel('S', 'LOC', 'Z', 0)
        mapdl.d('ALL', 'ALL')

        # Apply force on the free end of the beam
        mapdl.nsel('S', 'LOC', 'Z', self.total_len)
        mapdl.f('ALL', 'FY', -1000)  # Apply a downward force of 100 N

        # Perform solution
        mapdl.allsel(mute=True)
        mapdl.run('/SOLU')
        mapdl.antype('STATIC')
        try:
            mapdl.solve()
        except:
            print(f'warning: solving failure!')
            mapdl.exit()
            return None
        #if not mapdl.solution.converged:
        #    print(f'warning: solution not converged!')
        mapdl.finish()

        # Post-processing to find the maximum deformation
        result = mapdl.result
        # result.plot_principal_nodal_stress(
        #  0,
        #  "SEQV",
        #  lighting=False,
        #  background="w",
        #  show_edges=True,
        #  text_color="k",
        #  add_text=False,
        # )
        # nnum, stress = result.principal_nodal_stress(0)
        # von_mises = stress[:, -1]  # von-Mises stress is the right most column
        #
        # # Must use nanmax as stress is not computed at mid-side nodes
        # max_stress = np.nanmax(von_mises)
        # mask = result.mesh.nodes[:, 0] == self.total_len
        # far_field_stress = np.nanmean(von_mises[mask])
        # print("Far field von Mises stress: %e" % far_field_stress)
        # Fetch the maximum deformation result
        y_disp = mapdl.post_processing.nodal_displacement('Y')
        max_disp_y = np.max(y_disp)  # Use numpy to find the maximum value
        print(f"Maximum displacement in the Y-direction is {max_disp_y} meters.")
        return max_disp_y

    def mapping_func(self):
        self.modeling()
        return self.simulate()


# def mapping_func(paras):
#     modeling_instance = Modeling(paras)
#     modeling_instance.model()
#     return modeling_instance.simulate()

if __name__ == '__main__':
    paras = [0.5, 0.5]
    CAE(paras).mapping_func()