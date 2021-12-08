"""
Add version string to the navbar and footer.
"""
import dinf


def inject_version(app, config):
    v = dinf.__version__
    if v != "undefined":
        v_short = v.split("+")[0]
        config.html_theme_options["extra_navbar"] = f"dinf {v_short}"
        config.html_theme_options["extra_footer"] = f"dinf {v}"


def setup(app):
    app.connect("config-inited", inject_version)
