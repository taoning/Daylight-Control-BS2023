# Daylight-Control-BS2023
Evaluation of blinds control techniques for daylight and visual comfort in complex real-world conditions

## Optimization
Make sure to make a `Matrices` directory if it doesn't exists.
`python optimization`
A `opt.csv` file will be generated

## Rule-based
`python heuristic.py`
A `rb.csv` file will be generated.

## Evaluate control performance
Run `evalctrl.py` for each set of controls:
`python evalctrl.py opt.csv`
`python evalctrl.py rb.csv`
`python evalctrl.py hd.csv`
