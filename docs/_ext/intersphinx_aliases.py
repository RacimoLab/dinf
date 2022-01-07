"""
Add short aliases for internal namespaces in external packages.
"""
# Stolen from https://github.com/sphinx-doc/sphinx/issues/5603

intersphinx_aliases = {
    ("py:class", "cyvcf2.VCF"): ("py:class", "cyvcf2.cyvcf2.VCF"),
}


def add_intersphinx_aliases_to_inv(app):
    from sphinx.ext.intersphinx import InventoryAdapter

    inventories = InventoryAdapter(app.builder.env)

    for alias, target in intersphinx_aliases.items():
        alias_domain, alias_name = alias
        target_domain, target_name = target
        try:
            found = inventories.main_inventory[target_domain][target_name]
            try:
                inventories.main_inventory[alias_domain][alias_name] = found
            except KeyError:
                continue
        except KeyError:
            continue


def setup(app):
    app.connect("builder-inited", add_intersphinx_aliases_to_inv)
