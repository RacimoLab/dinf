(sec_cli)=
# CLI reference

Dinf provides two command line programs, `dinf` and `dinf-plot`.
The former provides subcommands for running analyses, while the latter
provides subcommands for making plots of various things.
When invoked with the `-h`/`--help` option, subcomands offer a concise
description and list the available options.
This help output is reproduced below.

Once Dinf is installed, the commands can be run by typing `dinf`,
`dinf-plot`, or `dinf-tabulate`. In addition, the commands can be run using
Python module hooks by typing `python -m dinf`, `python -m dinf.plot`,
or `python -m dinf.tabulate`.
The module hooks can be useful for running the commands from a
cloned git repository without requiring installation (e.g. during development).

(sec_cli_dinf)=
## Analysis/inference

```{program-output} python -m dinf -h
```

### `dinf check`

```{program-output} python -m dinf check -h
```

### `dinf train`

```{program-output} python -m dinf train -h
```

### `dinf predict`

```{program-output} python -m dinf predict -h
```

### `dinf smc`

```{program-output} python -m dinf smc -h
```

### `dinf mcmc-gan`

```{program-output} python -m dinf mcmc-gan -h
```

### `dinf pg-gan`

```{program-output} python -m dinf pg-gan -h
```


(sec_cli_dinf-plot)=
## Plotting results

```{program-output} python -m dinf.plot -h
```

### `dinf-plot metrics`

```{program-output} python -m dinf.plot metrics -h
```

### `dinf-plot features`

```{program-output} python -m dinf.plot features -h
```

### `dinf-plot hist`

```{program-output} python -m dinf.plot hist -h
```

### `dinf-plot hist2d`

```{program-output} python -m dinf.plot hist2d -h
```



(sec_cli_dinf-tabulate)=
## Tabulating results

```{program-output} python -m dinf.tabulate -h
```

### `dinf-tabulate metrics`

```{program-output} python -m dinf.tabulate metrics -h
```

### `dinf-tabulate data`

```{program-output} python -m dinf.tabulate data -h
```

### `dinf-tabulate quantiles`

```{program-output} python -m dinf.tabulate quantiles -h
```
