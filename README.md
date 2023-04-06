# Daylight-Control-BS2023
Evaluation of model-based control for daylight and visual comfort in a multi occupancy event space

## Optimization
Make sure to make a `Matrices` directory if it doesn't exists.
`python optimization`

## Rule-based
`python heuristic.py`
A `rb.csv` file will be generated.

## Evaluate control performance
Run `evalctrl.py` for each set of controls:
`python evalctrl.py opt.csv`
`python evalctrl.py rb.csv`
`python evalctrl.py hd.csv`
