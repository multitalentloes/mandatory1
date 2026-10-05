import numpy as np
import sympy as sp
from scipy import sparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import math

x, y, t = sp.symbols("x,y,t")


class Wave2D:
    """Class for solving the 2D wave equation"""


    def create_mesh(
        self, N: int, sparse: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return 2D mesh created using np.meshgrid

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        sparse : bool, optional
            Whether to create a sparse mesh or not. Default is False.
        Returns
        -------
        xij : 2D array
            The x-coordinates of the mesh
        yij : 2D array
            The y-coordinates of the mesh"""
        L = 1.0 # should just be on the unit square?
        xi = np.linspace(0, L, N + 1)
        xij, yij = np.meshgrid(xi, xi, indexing="ij", sparse=True)
        return xij, yij

    def D2(self, N: int) -> sparse.lil_matrix:
        """Return second order differentiation matrix

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Returns
        -------
        D : scipy sparse LIL matrix
            The second order differentiation matrix
        """
        return sparse.diags(([1.0, -2.0, 1.0]), (-1, 0, 1), (N, N), format="lil")

    @property
    def w(self):
        """Return the dispersion coefficient"""
        return self._w
        # raise NotImplementedError("The w property is not implemented yet.")

    def ue(self, mx: int, my: int) -> sp.Expr:
        """Return the exact standing wave

        Parameters
        ----------
        mx, my : int
            Parameters for the standing wave
        Returns
        -------
        ue : Sympy expression
            The exact solution as a Sympy expression in x, y and t
        """

        return sp.sin(mx * sp.pi * x) * sp.sin(my * sp.pi * y) * sp.cos(self.w * t)

    def initialize(self, N: int, mx: int, my: int) -> np.ndarray:
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        u = np.ndarray((N+1, N+1))
        xij, yij = self.create_mesh(N)
        return sp.lambdify((x, y, t), self.ue(mx, my))(xij, yij, 0)

    @property
    def dt(self) -> float:
        """Return the time step"""
        # I do not use this and instead just compute it from the cfl, c, and h in the main loop where
        # all those values are available!
        raise NotImplementedError("The dt property is not implemented yet.")

    def l2_error(self, u: np.ndarray, t0: float, mx, my) -> float:
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        N = u.shape[0]
        xij, yij = self.create_mesh(N-1)
        exact = sp.lambdify((x, y, t), self.ue(mx, my))(xij, yij, t0)

        dx = 1.0/(N-1)

        flat_diff = ((u-exact)**2).ravel()

        return (dx*dx*np.sum(flat_diff))**0.5

    def apply_bcs(self, u: np.ndarray):
        """Apply boundary conditions to the solution mesh function

        Parameters
        ----------
        u : array
            The solution mesh function
        """
        u[0,:]  = 0
        u[-1,:] = 0
        u[:,0]  = 0
        u[:,-1] = 0

        return u

    def __call__(
        self,
        N: int,
        Nt: int,
        cfl: float = 0.5,
        c: float = 1.0,
        mx: int = 3,
        my: int = 3,
        store_data: int = -1,
    ):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """ 

        self._w = c * math.pi * math.sqrt(mx ** 2 + my ** 2)
        dx = 1.0 / N
        dy = dx
        invdx2 = 1.0/(dx**2)

        dt = (cfl * dx) / c

        D_2 = self.D2(N+1)

        U_m1 = self.initialize(N, mx, my)
        U = U_m1 + ((c ** 2 * dt ** 2) / 2) * (invdx2 * D_2 @ U_m1 + invdx2*U_m1 @ D_2.T)
        self.apply_bcs(U)
        U_p1 = np.ndarray((N+1, N+1))

        all_sols = []
        if store_data > 0:
            all_sols = [U_m1, U]

        for t in range(Nt-1):
            # compute U_p1
            U_p1 = (c ** 2 * dt ** 2) * (invdx2 * D_2 @ U + invdx2*U @ D_2.T) + 2 * U - U_m1

            # handle boundary
            self.apply_bcs(U_p1)

            if store_data > 0:
                all_sols.append(U_p1)
            
            #rotate
            U_m1 = U
            U = U_p1
        
        if store_data > 0:
            return all_sols
        else:
            return (dx, self.l2_error(U, dt*Nt, mx, my))

    def convergence_rates(
        self, m: int = 4, cfl: float = 0.1, Nt: int = 10, mx: int = 3, my: int = 3
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for _ in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err)
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [
            np.log(E[i - 1] / E[i]) / np.log(h[i - 1] / h[i])
            for i in range(1, m, 1)
        ]
        return np.array(r), np.array(E), np.array(h)


class Wave2D_Neumann(Wave2D):
    def D2(self, N: int) -> sparse.lil_matrix:
        # raise NotImplementedError("The D2 method is not implemented yet.")
        diffmat =  sparse.diags(([1.0, -2.0, 1.0]), (-1, 0, 1), (N, N), format="lil")
        diffmat[0, 0] = -2
        diffmat[0, 1] = 2
        diffmat[N-1, N-2] = 2
        diffmat[N-1, N-1] = -2

        return diffmat

    def ue(self, mx: int, my: int) -> sp.Expr:
        return sp.cos(mx * sp.pi * x) * sp.cos(my * sp.pi * y) * sp.cos(self.w * t)

    def apply_bcs(self, u: np.ndarray):
        return # not needed, handled in the diff matrix!


def test_convergence_wave2d():
    sol = Wave2D()
    r, _, _ = sol.convergence_rates(m=5, mx=2, my=3)
    assert abs(r[-1] - 2) < 1e-2, r

def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, _, _ = solN.convergence_rates(mx=3, my=3)
    assert abs(r[-1] - 2) < 0.05


def test_exact_wave2d():
    sol = Wave2D()
    r, l2, _ = sol.convergence_rates(m=5, mx=2,my=2, cfl=1/math.sqrt(2)) # since mx=my dispersion is exact!
    assert max(l2) < 10e-12
    sol = Wave2D_Neumann()
    r2, l22, _ = sol.convergence_rates(m=5, mx=2,my=2, cfl=1/math.sqrt(2)) # since mx=my dispersion is exact!
    
    assert max(l22) < 10e-12

    return # successful

def make_gifs():
    sols = Wave2D()(32, 100, cfl=1/math.sqrt(2), mx=3, my=2, store_data=1)
    sols = np.asarray(sols)

    interval=1
    fps=10

    fig, ax = plt.subplots()

    # Keep the color scale fixed over the whole animation
    vmin = sols.min()
    vmax = sols.max()

    im = ax.imshow(
        sols[0],
        origin="lower",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        animated=True,
    )

    fig.colorbar(im, ax=ax, label="u")
    title = ax.set_title("timestep 0")

    def update(frame):
        im.set_data(sols[frame])
        title.set_text(f"timestep {frame}")
        return im, title

    anim = FuncAnimation(
        fig,
        update,
        frames=len(sols),
        interval=interval,
        blit=True,
    )

    anim.save("dirichlet.gif", writer=PillowWriter(fps=fps))
    plt.close(fig)


if __name__ == "__main__":
    test_convergence_wave2d()
    test_convergence_wave2d_neumann()
    test_exact_wave2d()
    # make_gifs()
