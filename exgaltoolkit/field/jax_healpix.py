from functools import partial
import jax
import jax.numpy as jnp 

# may need to update for nside>8192
jxpx_int = jnp.int32

@jax.jit
def imodulo64(v1, v2):
    v = jxpx_int(v1%v2)
    return jnp.where(v >= 0, v, v + v2)

@jax.jit
def fmodulo(v1, v2):
    temp = jnp.mod(v1 , v2) + v2
    return jnp.where(v1 >= 0., jnp.where(v1 < v2, v1, jnp.mod(v1,v2)), jnp.where(temp == 0., 0., temp))

@partial(jax.jit, static_argnames=['nside'])
def ang2pix_ring_zphi(nside, z, s, phi):
    za = jnp.abs(z)
    tt = fmodulo(phi, 2*jnp.pi) * (2. / jnp.pi)

    def zt2pix_ring_eq(nside, z, tt):
        temp1 = nside * (tt + 0.5)
        temp2 = nside * z * 0.75 

        jp = jxpx_int(temp1 - temp2)
        jm = jxpx_int(temp1 + temp2)
        ir = nside + 1 + jp - jm

        kshift = 1 - (ir & 1)
        ip = imodulo64(jxpx_int((jp + jm - nside + kshift + 1) // 2), 4*nside)

        return jxpx_int(nside * (nside - 1) * 2 + (ir - 1) * 4*nside + ip)
    
    def zt2pix_ring_pol(nside, z, za, tt, s):
        tp = tt - jnp.int16(tt)
        temp = jnp.where(s > -2., nside * s / jnp.sqrt((1. + za) / 3.), nside * jnp.sqrt(3. * ( 1. - za)))

        jp = jxpx_int(tp * temp)
        jm = jxpx_int((1. - tp) * temp)

        ir = jxpx_int(jp + jm + 1)
        ip = imodulo64(jxpx_int(tt * ir), 4*ir)

        return jxpx_int(jnp.where(z > 0, 2 * ir * (ir - 1) + ip, 12 * nside * nside - 2 * ir * (ir + 1) + ip))

    return jnp.where(za <= (2./3.), zt2pix_ring_eq(nside, z, tt), zt2pix_ring_pol(nside, z, za, tt, s))


@partial(jax.jit, static_argnames=['nside'])
def vec2pix(nside, x, y, z):
    
    vlen = jnp.sqrt(x*x + y*y + z*z)
    cth = z / vlen
    sth = jnp.where(jnp.abs(cth) > 0.99, jnp.sqrt(x*x + y*y) / vlen, -5.)
    return ang2pix_ring_zphi(nside, cth, sth, jnp.arctan2(y, x))


