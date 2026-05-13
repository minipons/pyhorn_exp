# pyhorn-exp

Experimental packages for the pyhorn ecosystem: ML optimization, waveguide analysis, and CAD integration tools.

## Packages

| Package | Description |
|---------|-------------|
| `pyhorn_ml` | Machine learning optimization — differential evolution for horn geometry synthesis |
| `pyhorn_wavefront` | 2D wave propagation FDTD simulation with PML absorbing boundaries |
| `pyhorn_fold` | Horn folding pattern generator |
| `pyhorn_segment` | Horn segmentation and medial-axis tools |
| `pyhorn_registry` | Driver and project registry with YAML persistence |
| `integrations.onshape` | Onshape CAD integration — import/export horn geometry from Onshape |

## ML Optimization

```bash
pyhorn optimize --driver drivers/FE166NV2.yaml --target-fs 50 --target-qts 0.28
```

## Wavefront Simulation

```bash
pyhorn wavefront -h projects/bkhiro.yaml --freq 200 --duration 50
```

## Onshape Integration

Export a horn from Onshape as JSON, then:
```bash
pyhorn auto-segment -i onshape_export.json -o horn.yaml --n-segments 20
```

## Dependencies

- pyhorn (core)
- numpy, scipy, scikit-learn, scikit-optimize
- shapely, networkx, matplotlib, pyyaml, typer
