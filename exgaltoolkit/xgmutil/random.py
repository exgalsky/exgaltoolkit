import jax.numpy as jnp 
import jax.random as rdm 
import jax

import xgmutil.stream as stream

cmb_dict = {'cmb_tlm': 0, 'cmb_elm': 1, 'cmb_blm': 2}
lens_dict = {'phi_lm': 3}
# dust_list = [[4, 'dust_tlm'], [5, 'dust_elm'], [6, 'dust_blm']]
# sync_list = [[7, 'sync_tlm'], [8, 'sync_elm'], [9, 'sync_blm']]
ic_dict = {'ic_grid': 32}

predefined_components = cmb_dict | lens_dict | ic_dict

class RNG_component:
    def __init__(self, component_name, stream_id, stream_key, **kwargs):
        self.component     = component_name
        self.component_id  = stream_id
        self.component_key = stream_key
        self.nsub          = kwargs.get('nsub',1024**3)
        self.dtype         = kwargs.get('dtype',jnp.float32)
        _force_no_gpu      = kwargs.get('force_no_gpu',False)

        self.device        = jax.default_backend()
        if _force_no_gpu: self.device = 'cpu'

        self._dtype_stream = self.dtype
        if self.dtype == jnp.float32: 
            self._dtype_stream = jnp.float64
        elif self.dtype == jnp.complex64: 
            self._dtype_stream = jnp.complex128

        self.stream = stream.Stream(force_no_gpu=_force_no_gpu, seedkey=self.component_key, nsub=self.nsub, dtype=self._dtype_stream)

    def generate(self, **kwargs):
        mc  = kwargs.get('mc', 0)
        self.stream.set_seedkey(mc)

        if self.dtype != self._dtype_stream: return (self.stream.generate(**kwargs)).astype(self.dtype)
        return self.stream.generate(**kwargs)

class RNG_manager:
    def __init__(self, seed = 57885161):
    # Mersenne exponents no 48. Ref: https://oeis.org/A000043 used as seed
    # Do we hard code this here, or have it defined elsewhere and read here?
        self.seed = seed
        self.n_components = 512

        self.master_key  = rdm.PRNGKey(self.seed)
        self.key_list    = rdm.split(self.master_key, num=self.n_components)
        self.stream_no   = jnp.arange(0, self.n_components, dtype=jnp.int16)
        self.stream_id   = [f"{self.seed}-{seq_no:{0}>3}" for seq_no in self.stream_no]
        self.registry    = [None for _ in self.stream_no]

        for key in predefined_components:
            self.registry[predefined_components[key]] = key  

    def _register_component(self, component_name, component_type='extragalactic'):
        unassigned_idx = jnp.array([i for i, val in enumerate(self.registry) if val == None])
        unassigned_stream = self.stream_no[unassigned_idx]
        self.registry[jnp.min(unassigned_stream[unassigned_stream >= 128])] = component_name.lower()

    def setup_stream(self, component_name, **kwargs):
        if not(component_name in self.registry): self._register_component(component_name)
        stream_no = self.registry.index(component_name)
        return RNG_component(component_name, self.stream_id[stream_no], self.key_list[stream_no], **kwargs)
            
    

