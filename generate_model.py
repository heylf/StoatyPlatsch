import numpy

from lmfit import models

ALL_POSSIBLE_DIST = ['GaussianModel', 'LorentzianModel', 'VoigtModel', 'SkewedGaussianModel', 'SkewedVoigtModel', 'DonaichModel']

def generate_model(spec, min_peak_width, max_peak_width):
    composite_model = None
    params = None
    y = spec['y']
    y_max = numpy.max(y)
    for i, basis_func in enumerate(spec['model']):
        prefix = f'm{i}_'
        model = getattr(models, basis_func['type'])(prefix=prefix)
        if basis_func['type'] in ALL_POSSIBLE_DIST:
            model.set_param_hint('sigma', min=min_peak_width, max=max_peak_width)
            model.set_param_hint('center', vary=False)
            model.set_param_hint('height', min=1e-6, max=1.1*y_max)
            model.set_param_hint('amplitude', min=1e-6, max=1.1*y_max)
        else:
            raise NotImplemented(f'model {basis_func["type"]} not implemented yet')
        model_params = model.make_params(**basis_func.get('params', {}))
        if params is None:
            params = model_params
        else:
            params.update(model_params)
        if composite_model is None:
            composite_model = model
        else:
            composite_model = composite_model + model
    return composite_model, params