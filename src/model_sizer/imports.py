import importlib 
import importlib.metadata

def _is_package_available(pkg_name):
    # Check we're not importing a "pkg_name" directory somewhere but the actual library by trying to grab the version
    package_exists = importlib.util.find_spec(pkg_name) is not None
    if package_exists:
        try:
            _ = importlib.metadata.metadata(pkg_name)
            return True
        except importlib.metadata.PackageNotFoundError:
            return False
        
def is_transformers_available():
    return _is_package_available("transformers")

def is_diffusers_available():
    return _is_package_available("diffusers")

def is_timm_available():
    return _is_package_available("timm")