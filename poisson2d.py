import numpy as np
import sympy as sp
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from poisson import Poisson

x, y = sp.symbols("x,y")

# Below we create a solver that reuses some of the implementation from
# the 1D solver in poisson.py.


class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), x, y in [0, L] x [0, L]

    with Dirichlet boundary conditions.
    """

    def __init__(self, L: float):
        self.p = Poisson(L)  # we can reuse some of the code from the 1D case

    def create_mesh(self, N: int) -> tuple[np.ndarray, np.ndarray]:
        """Return a 2D Cartesian mesh

        Parameters
        ----------
        N : int
            The number of uniform intervals in both x and y directions
        Returns
        -------
        xij : 2D array
            The x-coordinates of the mesh
        yij : 2D array
            The y-coordinates of the mesh
        """
        xi = self.p.create_mesh(N)
        xij, yij = np.meshgrid(xi, xi, indexing="ij", sparse=True)
        return xij, yij

    def laplace(self, N: int) -> sparse.lil_matrix:
        """Return a vectorized Laplace operator

        Parameters
        ----------
        N : int
            The number of uniform intervals in both x and y directions

        Returns
        -------
        A : scipy sparse LIL matrix
            The vectorized Laplace operator
        """
        return sparse.diags(([1.0, -2.0, 1.0]), (-1, 0, 1), (N, N), format="lil")

    def assemble(
        self, N: int, f: sp.Expr, ue: sp.Expr
    ) -> tuple[sparse.csr_matrix, np.ndarray]:
        """Return assembled coefficient matrix A and right hand side vector b

        Parameters
        ----------
        Nx : int
            The number of uniform intervals in both x and y directions
        f : Sympy expression
            The right hand side as a Sympy expression in x and y
        ue : Sympy expression
            The exact solution as a Sympy expression in x and y

        Returns
        -------
        A : scipy sparse CSR matrix
            Coefficient matrix
        b : 1D array
            Right hand side vector

        Note
        ----
        Compute the Kronecker product of the 1D Laplace operator with itself
        to create the 2D Laplace operator. Then, assemble the right-hand side
        vector b by evaluating the function f at the mesh points and applying
        Dirichlet boundary conditions using the exact solution ue.

        """

        # create A = (kron(Dx,I)+kron(I,Dy))
        D = self.laplace(N+1) # differentiation matrix
        D_x = ((1.0/(self.p.L/N))**2) * D
        A = sparse.kron(D_x, sparse.eye(N+1), format="lil")
        A += sparse.kron(sparse.eye(N+1), D_x)

        # create b (ravel(F)) where f is the method of manufactured solution evaluated on the domain
        xij, yij = self.create_mesh(N)
        F = self.meshfunction(f, xij, yij)
        b = np.ravel(F)

        # handle boundary conditions:
        bnds = self.get_boundary_indices(N)
        
        # b[bnds] = 0 # would be true for dirichlet = 0 condition, but we are using manufactured solutions!
        Ue = self.meshfunction(ue, xij, yij)
        b[bnds] = Ue.ravel()[bnds]

        for idx in bnds:
            # we have to ident row = idx
            A[idx, :] = 0
            A[idx, idx] = 1

        A = A.tocsr()

        return (A, b)

        # raise NotImplementedError("The assemble method is not implemented yet.")

    def meshfunction(self, u: sp.Expr, xij: np.ndarray, yij: np.ndarray) -> np.ndarray:
        """Return Sympy function as mesh function

        Parameters
        ----------
        u : Sympy function

        Returns
        -------
        array - The input function as a mesh function
        """

        return sp.lambdify((x, y), u)(xij, yij)

    def get_boundary_indices(self, N: int) -> np.ndarray:
        """Return indices of vectorized matrix that belongs to the boundary"""
        bbox = np.ones((N+1, N+1))
        bbox[0,:] = 0 # zero out top row
        bbox[N,:] = 0 # zero out bottom row
        bbox[:,0] = 0 # zero out left side
        bbox[:,N] = 0 # zero out right side
        
        return np.where(bbox.ravel() == 0)[0]

    def l2_error(self, u: np.ndarray, ue: sp.Expr) -> float:
        """Return l2-error

        Parameters
        ----------
        u : array
            The numerical solution (mesh function)
        ue : Sympy expression
            The exact solution

        Returns
        -------
        float - The l2-error

        """
        
        # u is indexable, but ue is continuous, how should i "align" them when computing the
        # diff mesh function which I can take the norm of?

        N = u.shape[0]


        xij, yij = self.create_mesh(N-1)


        dxdy = (self.p.L / (u.shape[0] - 1)) ** 2 # dx = dy and dx*dy will be multiplied with error squared at each point in the mesh

        mf = self.meshfunction(ue, xij, yij)

        flat_diff = ((u-mf)**2).ravel()

        return (dxdy*np.sum(flat_diff))**0.5



    def __call__(self, N: int, ue: sp.Expr) -> np.ndarray:
        """Solve Poisson's equation with a given manufactured solution

        Parameters
        ----------
        Nx : int
            The number of uniform intervals in both x and y directions
        ue : Sympy expression
            The exact solution

        Returns
        -------
        The solution as a Numpy array

        """
        A, b = self.assemble(N, sp.diff(ue, x, 2) + sp.diff(ue, y, 2), ue)
        return sparse_linalg.spsolve(A, b.ravel()).reshape((N + 1, N + 1))

    def convergence_rates(self, ue: sp.Expr, m: int = 6): # m should be 6, temporary change for profiling
        E = []
        h = []
        N0 = 8
        for _ in range(m):
            u = self(N0, ue)
            E.append(self.l2_error(u, ue))
            h.append(self.p.L / N0)
            N0 *= 2
            print(f"Done with {N0} spatial step size")
        r = [np.log(E[i - 1] / E[i]) / np.log(h[i - 1] / h[i]) for i in range(1, m, 1)]
        print(E)
        return r, np.array(E), np.array(h)

    def eval(self, U: np.ndarray, x: float, y: float) -> float:
        """Return u(x, y)

        Parameters
        ----------
        x, y : numbers
            The coordinates for evaluation

        Returns
        -------
        The value of u(x, y)

        """
        h = self.p.L / (U.shape[1] - 1)
        N = U.shape[0]
        assert U.shape[0] == U.shape[1] # cartesian discretization with hx = hy of unit square

        xbefore = int(x // h)
        ybefore = int(y // h)

        print(xbefore, ybefore)


        def lx(xs, x: float, xj: int, dx: float):
            ans = 1
            for i in xs:
                if i != xj:
                    ans *= (x-i*dx)/(xj*dx - i*dx)
            return ans


        ans = 0
        for i in [xbefore, xbefore+1]:
            for j in [ybefore, ybefore+1]:
                ans += U[i,j] * lx([xbefore, xbefore+1], x, i, h) * lx([ybefore, ybefore+1], y, j, h)

        return ans
        # raise NotImplementedError("The eval method is not implemented yet.")


def test_convergence_poisson2d():
    # This exact solution is NOT zero on the entire boundary
    ue = sp.exp(sp.cos(4 * sp.pi * x) * sp.sin(2 * sp.pi * y))
    sol = Poisson2D(1)
    r, _, _ = sol.convergence_rates(ue)
    print(r)
    assert abs(r[-1] - 2) < 1e-2


def test_interpolation():
    ue = sp.exp(sp.cos(4 * sp.pi * x) * sp.sin(2 * sp.pi * y))
    sol = Poisson2D(1)
    N = 100
    U = sol(N, ue)
    h = sol.p.L / N
    
    # print(sol.eval(U, 1 - 3*h / 2, 1 - 3*h / 2), ue.subs({x: 1 - 3*h / 2, y: 1 - 3*h / 2}).n())
    assert abs(sol.eval(U, 0.52, 0.63) - ue.subs({x: 0.52, y: 0.63}).n()) < 1e-3
    assert abs(sol.eval(U, h / 2, 1 - h / 2) - ue.subs({x: h / 2, y: 1 - h / 2}).n()) < 1e-3

def test_laplace():
    p = Poisson2D(10)
    A = p.laplace(40)

    # check that [1, -2, 1] pattern holds!
    for row in range(40):
        for col in range(40):
            if col and col + 1 == row:
                assert A[row, col] == 1.0
            if row == col:
                assert A[row, col] == -2.0
            if col < 39 and col - 1 == row:
                assert A[row, col] == 1.0

def test_boundary_indices():
    p = Poisson2D(10)
    bbox = p.get_boundary_indices(10)

def test_symbolic_mesh_function():
    p = Poisson2D(10)

    u = 2*x + y
    xij, yij = p.create_mesh(10)

    m = p.meshfunction(u, xij, yij)

    for row in range(10):
        for col in range(10):
            assert m[row, col] == 2*row + col

if __name__ == "__main__":
    test_laplace()
    test_boundary_indices()
    test_symbolic_mesh_function()
    test_convergence_poisson2d()
    test_interpolation()
    print("All tests passed!")
