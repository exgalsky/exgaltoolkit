class CosmologyInterface:
    '''CosmologyInterface'''
    def __init__(self, **kwargs):
        self.h      = 0.7
        self.omegam = 0.276
        self.omegak = 0.0
        self.omegal = 1 - self.omegam - self.omegak
        return

    def get_pspec(self):
        import numpy as np
        import jax.numpy as jnp

        # placeholder for power spectrum interface
        from importlib.resources import files
        pkfile = files("exgaltoolkit.data").joinpath("camb_40107036_matterpower.dat")
        k, pk = np.loadtxt(pkfile,usecols=(0,1),unpack=True)
        
        # create dictionary wrapper for power spectrum
        self.pspec = {'k': jnp.asarray(k), 'pofk': jnp.asarray(pk)}

    def get_growth(self):
        import jax
        import jax.numpy as jnp

        # placeholder for dynamics interface
        z   = np.linspace(0,100,1000)

        # hubble parameter
        Hofz = 100*h**2*jnp.sqrt(omegal + omegak*(1+z)**2 + omegam*(1+z)**3)

        # linear growth factor using w=-1 fitting formulae from Carrol, Press & Turner (1992)
        w  = -1
        x  = 1+z
        x2 = x*x
        x3 = x*x*x
        x3w= x3**w

        omega  = omegam*x3/( omegam*x3 + (1-omegam-omegal)*x2 + omegal )
        Lambda = omegal*x3*x3w/(omegam*x3+(1-omegam-omegal)*x2+omegal*x3*x3w)

        g  = 2.5*omega /(pow(omega ,4./7.)-Lambda+(1+omega /2)*(1+Lambda/70))
        g0 = 2.5*omegam/(pow(omegam,4./7.)-omegal+(1+omegam/2)*(1+omegal/70))
        D1 = (g/x)/g0

        # 2nd order growth factor approximation from Bernardeau et al. (2001)
        D2 =  -3/7 * omegam**(-1/143) * D**2

        # f derivatives by autograd of growth factors
        D1ofa = lambda a: jnp.interp(a,1/(1+z),D1)
        D2ofa = lambda a: jnp.interp(a,1/(1+z),D2)

        f1ofz = lambda z: jax.grad(D1ofa)(1/(1+z)) / D1ofa(1/(1+z)) / (1+z)
        f2ofz = lambda z: jax.grad(D2ofa)(1/(1+z)) / D2ofa(1/(1+z)) / (1+z)

        f1 = f1ofz(z)
        f2 = f2ofz(z)

        self.growth = jnp.asarray([z,H,D1,D2,f1,f2])

