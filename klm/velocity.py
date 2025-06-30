import numpy as np

class VelocityModel():
	def __init__(self, rc_type='arctan') -> None:
		self._select_rotation_curve(rc_type)

	def build_vfield(self, pars, Xgrid, Ygrid):
		v_0 = pars['v_0']
		vcirc = pars['vcirc']
		cosi = pars['cosi']
		vscale = pars['vscale']
		v_outer = pars['v_outer']

		## Offset params
		dx_vel = pars['dx_vel']
		dy_vel = pars['dy_vel']

		r_hl = pars['r_hl_disk']
		sini = (1-cosi**2)**0.5

		Xgrid_offset = Xgrid - dx_vel * r_hl
		Ygrid_offset = Ygrid - dy_vel * r_hl

		theta = np.angle(Xgrid_offset + 1j*Ygrid_offset)

		R = (Xgrid_offset**2 + Ygrid_offset**2)**0.5

		rotation_curve = vcirc *  self._rotation_curve_model(R / vscale) + v_outer*R

		Vfield = v_0 +  sini * np.cos(theta) * rotation_curve

		return Vfield

	def _select_rotation_curve(self, rc_type):
		if rc_type == 'arctan':
			self._rotation_curve_model = lambda x: 2/np.pi*np.arctan(x)

		elif rc_type == 'tanh':
			self._rotation_curve_model = np.tanh
