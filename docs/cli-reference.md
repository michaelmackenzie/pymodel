# CLI reference

Generated from `--help` for roomodel; the other backends differ only in their
backend-specific options. Regenerate with `scripts/make_cli_reference.sh`.

## `build`

```
usage: pymodel roomodel build [-h] [--mass MASS] [--rmin RMIN] [--rmax RMAX]
                              [--bin-integration {center,integral}]
                              [--set-parameters SET_PARAMETERS]
                              [--freeze-parameters FREEZE_PARAMETERS]
                              [--freeze-nuisance-groups FREEZE_NUISANCE_GROUPS]
                              [--set-parameter-ranges SET_PARAMETER_RANGES] [--seed SEED]
                              [--strategy {0,1,2}] [--tolerance TOLERANCE] [--output OUTPUT]
                              [--plot] [--plot-dir PLOT_DIR] [--verbose] [--bundle BUNDLE]
                              input

options:
  -h, --help            show this help message and exit
  --bundle BUNDLE       output bundle path (default model.json)

model:
  input                 Combine datacard, or a model bundle (.json) written by 'build'
  --mass MASS           value substituted for $MASS in shapes lines
  --rmin RMIN           POI lower bound (default 0)
  --rmax RMAX           POI upper bound (default 20, as in Combine)
  --bin-integration {center,integral}
                        how parametric pdfs are evaluated on binned data: bin centre x width
                        (Combine/RooFit, default) or exact bin integrals
  --set-parameters SET_PARAMETERS
                        name=value,... (initial/fixed values)
  --freeze-parameters FREEZE_PARAMETERS
                        name,... parameters to fix at their values
  --freeze-nuisance-groups FREEZE_NUISANCE_GROUPS
                        group,... datacard groups to freeze
  --set-parameter-ranges SET_PARAMETER_RANGES
                        name=lo:hi,...

fit / output:
  --seed SEED           random seed (default 123456, as in Combine)
  --strategy {0,1,2}    Minuit strategy
  --tolerance TOLERANCE
                        Minuit tolerance (EDM goal 0.002*tol*0.5)
  --output OUTPUT, -o OUTPUT
                        result JSON (default pymodel_<command>.json)
  --plot                write plots for this command
  --plot-dir PLOT_DIR
  --verbose, -v
```

## `inspect`

```
usage: pymodel roomodel inspect [-h] [--mass MASS] [--rmin RMIN] [--rmax RMAX]
                                [--bin-integration {center,integral}]
                                [--set-parameters SET_PARAMETERS]
                                [--freeze-parameters FREEZE_PARAMETERS]
                                [--freeze-nuisance-groups FREEZE_NUISANCE_GROUPS]
                                [--set-parameter-ranges SET_PARAMETER_RANGES] [--seed SEED]
                                [--strategy {0,1,2}] [--tolerance TOLERANCE] [--output OUTPUT]
                                [--plot] [--plot-dir PLOT_DIR] [--verbose]
                                input

options:
  -h, --help            show this help message and exit

model:
  input                 Combine datacard, or a model bundle (.json) written by 'build'
  --mass MASS           value substituted for $MASS in shapes lines
  --rmin RMIN           POI lower bound (default 0)
  --rmax RMAX           POI upper bound (default 20, as in Combine)
  --bin-integration {center,integral}
                        how parametric pdfs are evaluated on binned data: bin centre x width
                        (Combine/RooFit, default) or exact bin integrals
  --set-parameters SET_PARAMETERS
                        name=value,... (initial/fixed values)
  --freeze-parameters FREEZE_PARAMETERS
                        name,... parameters to fix at their values
  --freeze-nuisance-groups FREEZE_NUISANCE_GROUPS
                        group,... datacard groups to freeze
  --set-parameter-ranges SET_PARAMETER_RANGES
                        name=lo:hi,...

fit / output:
  --seed SEED           random seed (default 123456, as in Combine)
  --strategy {0,1,2}    Minuit strategy
  --tolerance TOLERANCE
                        Minuit tolerance (EDM goal 0.002*tol*0.5)
  --output OUTPUT, -o OUTPUT
                        result JSON (default pymodel_<command>.json)
  --plot                write plots for this command
  --plot-dir PLOT_DIR
  --verbose, -v
```

## `nll`

```
usage: pymodel roomodel nll [-h] [--mass MASS] [--rmin RMIN] [--rmax RMAX]
                            [--bin-integration {center,integral}]
                            [--set-parameters SET_PARAMETERS]
                            [--freeze-parameters FREEZE_PARAMETERS]
                            [--freeze-nuisance-groups FREEZE_NUISANCE_GROUPS]
                            [--set-parameter-ranges SET_PARAMETER_RANGES] [--seed SEED]
                            [--strategy {0,1,2}] [--tolerance TOLERANCE] [--output OUTPUT]
                            [--plot] [--plot-dir PLOT_DIR] [--verbose] [--at AT] [--toys TOYS]
                            [--expect-signal EXPECT_SIGNAL] [--toys-frequentist]
                            [--bypass-frequentist-fit] [--toys-no-systematics]
                            [--toys-file TOYS_FILE]
                            input

options:
  -h, --help            show this help message and exit
  --at AT               name=value,... parameter point (default: nominal)

model:
  input                 Combine datacard, or a model bundle (.json) written by 'build'
  --mass MASS           value substituted for $MASS in shapes lines
  --rmin RMIN           POI lower bound (default 0)
  --rmax RMAX           POI upper bound (default 20, as in Combine)
  --bin-integration {center,integral}
                        how parametric pdfs are evaluated on binned data: bin centre x width
                        (Combine/RooFit, default) or exact bin integrals
  --set-parameters SET_PARAMETERS
                        name=value,... (initial/fixed values)
  --freeze-parameters FREEZE_PARAMETERS
                        name,... parameters to fix at their values
  --freeze-nuisance-groups FREEZE_NUISANCE_GROUPS
                        group,... datacard groups to freeze
  --set-parameter-ranges SET_PARAMETER_RANGES
                        name=lo:hi,...

fit / output:
  --seed SEED           random seed (default 123456, as in Combine)
  --strategy {0,1,2}    Minuit strategy
  --tolerance TOLERANCE
                        Minuit tolerance (EDM goal 0.002*tol*0.5)
  --output OUTPUT, -o OUTPUT
                        result JSON (default pymodel_<command>.json)
  --plot                write plots for this command
  --plot-dir PLOT_DIR
  --verbose, -v

toys (Combine semantics):
  --toys TOYS, -t TOYS  number of toys (-1: Asimov dataset)
  --expect-signal EXPECT_SIGNAL
                        r used to generate toys (default 0)
  --toys-frequentist    fit nuisances to data and randomise global observables (Combine
                        --toysFrequentist)
  --bypass-frequentist-fit
                        frequentist toys around the pre-fit nuisance values
  --toys-no-systematics
                        do not randomise nuisances
  --toys-file TOYS_FILE
                        read datasets saved by 'generate' instead of generating
```

## `fit`

```
usage: pymodel roomodel fit [-h] [--mass MASS] [--rmin RMIN] [--rmax RMAX]
                            [--bin-integration {center,integral}]
                            [--set-parameters SET_PARAMETERS]
                            [--freeze-parameters FREEZE_PARAMETERS]
                            [--freeze-nuisance-groups FREEZE_NUISANCE_GROUPS]
                            [--set-parameter-ranges SET_PARAMETER_RANGES] [--seed SEED]
                            [--strategy {0,1,2}] [--tolerance TOLERANCE] [--output OUTPUT]
                            [--plot] [--plot-dir PLOT_DIR] [--verbose] [--toys TOYS]
                            [--expect-signal EXPECT_SIGNAL] [--toys-frequentist]
                            [--bypass-frequentist-fit] [--toys-no-systematics]
                            [--toys-file TOYS_FILE] [--minos MINOS] [--fix-r FIX_R]
                            input

options:
  -h, --help            show this help message and exit
  --minos MINOS         comma-separated parameters for MINOS errors ('all' for all)
  --fix-r FIX_R         fit with r fixed to this value

model:
  input                 Combine datacard, or a model bundle (.json) written by 'build'
  --mass MASS           value substituted for $MASS in shapes lines
  --rmin RMIN           POI lower bound (default 0)
  --rmax RMAX           POI upper bound (default 20, as in Combine)
  --bin-integration {center,integral}
                        how parametric pdfs are evaluated on binned data: bin centre x width
                        (Combine/RooFit, default) or exact bin integrals
  --set-parameters SET_PARAMETERS
                        name=value,... (initial/fixed values)
  --freeze-parameters FREEZE_PARAMETERS
                        name,... parameters to fix at their values
  --freeze-nuisance-groups FREEZE_NUISANCE_GROUPS
                        group,... datacard groups to freeze
  --set-parameter-ranges SET_PARAMETER_RANGES
                        name=lo:hi,...

fit / output:
  --seed SEED           random seed (default 123456, as in Combine)
  --strategy {0,1,2}    Minuit strategy
  --tolerance TOLERANCE
                        Minuit tolerance (EDM goal 0.002*tol*0.5)
  --output OUTPUT, -o OUTPUT
                        result JSON (default pymodel_<command>.json)
  --plot                write plots for this command
  --plot-dir PLOT_DIR
  --verbose, -v

toys (Combine semantics):
  --toys TOYS, -t TOYS  number of toys (-1: Asimov dataset)
  --expect-signal EXPECT_SIGNAL
                        r used to generate toys (default 0)
  --toys-frequentist    fit nuisances to data and randomise global observables (Combine
                        --toysFrequentist)
  --bypass-frequentist-fit
                        frequentist toys around the pre-fit nuisance values
  --toys-no-systematics
                        do not randomise nuisances
  --toys-file TOYS_FILE
                        read datasets saved by 'generate' instead of generating
```

## `scan`

```
usage: pymodel roomodel scan [-h] [--mass MASS] [--rmin RMIN] [--rmax RMAX]
                             [--bin-integration {center,integral}]
                             [--set-parameters SET_PARAMETERS]
                             [--freeze-parameters FREEZE_PARAMETERS]
                             [--freeze-nuisance-groups FREEZE_NUISANCE_GROUPS]
                             [--set-parameter-ranges SET_PARAMETER_RANGES] [--seed SEED]
                             [--strategy {0,1,2}] [--tolerance TOLERANCE] [--output OUTPUT]
                             [--plot] [--plot-dir PLOT_DIR] [--verbose] [--param PARAM]
                             [--points POINTS] [--range RANGE] [--toys TOYS]
                             [--expect-signal EXPECT_SIGNAL] [--toys-frequentist]
                             [--bypass-frequentist-fit] [--toys-no-systematics]
                             [--toys-file TOYS_FILE]
                             input

options:
  -h, --help            show this help message and exit
  --param PARAM
  --points POINTS
  --range RANGE         lo:hi (default: POI range, or the parameter range)

model:
  input                 Combine datacard, or a model bundle (.json) written by 'build'
  --mass MASS           value substituted for $MASS in shapes lines
  --rmin RMIN           POI lower bound (default 0)
  --rmax RMAX           POI upper bound (default 20, as in Combine)
  --bin-integration {center,integral}
                        how parametric pdfs are evaluated on binned data: bin centre x width
                        (Combine/RooFit, default) or exact bin integrals
  --set-parameters SET_PARAMETERS
                        name=value,... (initial/fixed values)
  --freeze-parameters FREEZE_PARAMETERS
                        name,... parameters to fix at their values
  --freeze-nuisance-groups FREEZE_NUISANCE_GROUPS
                        group,... datacard groups to freeze
  --set-parameter-ranges SET_PARAMETER_RANGES
                        name=lo:hi,...

fit / output:
  --seed SEED           random seed (default 123456, as in Combine)
  --strategy {0,1,2}    Minuit strategy
  --tolerance TOLERANCE
                        Minuit tolerance (EDM goal 0.002*tol*0.5)
  --output OUTPUT, -o OUTPUT
                        result JSON (default pymodel_<command>.json)
  --plot                write plots for this command
  --plot-dir PLOT_DIR
  --verbose, -v

toys (Combine semantics):
  --toys TOYS, -t TOYS  number of toys (-1: Asimov dataset)
  --expect-signal EXPECT_SIGNAL
                        r used to generate toys (default 0)
  --toys-frequentist    fit nuisances to data and randomise global observables (Combine
                        --toysFrequentist)
  --bypass-frequentist-fit
                        frequentist toys around the pre-fit nuisance values
  --toys-no-systematics
                        do not randomise nuisances
  --toys-file TOYS_FILE
                        read datasets saved by 'generate' instead of generating
```

## `limit`

```
usage: pymodel roomodel limit [-h] [--mass MASS] [--rmin RMIN] [--rmax RMAX]
                              [--bin-integration {center,integral}]
                              [--set-parameters SET_PARAMETERS]
                              [--freeze-parameters FREEZE_PARAMETERS]
                              [--freeze-nuisance-groups FREEZE_NUISANCE_GROUPS]
                              [--set-parameter-ranges SET_PARAMETER_RANGES] [--seed SEED]
                              [--strategy {0,1,2}] [--tolerance TOLERANCE] [--output OUTPUT]
                              [--plot] [--plot-dir PLOT_DIR] [--verbose]
                              [--method {asymptotic,toys}] [--cl CL]
                              [--run {both,observed,expected,blind}] [--grid GRID]
                              [--toys-per-point TOYS_PER_POINT] [--refine REFINE]
                              [--bypass-frequentist-fit] [--toys-file TOYS_FILE]
                              [--toy-index TOY_INDEX]
                              input

options:
  -h, --help            show this help message and exit
  --method {asymptotic,toys}
  --cl CL
  --run {both,observed,expected,blind}
                        blind: expected only, from a pre-fit Asimov dataset
  --grid GRID           toys: r grid 'lo:hi:n' or list (default: around the asymptotic limit)
  --toys-per-point TOYS_PER_POINT
  --refine REFINE       toys: bisection points added around the crossing
  --bypass-frequentist-fit
  --toys-file TOYS_FILE
                        use a saved dataset as the observed data
  --toy-index TOY_INDEX

model:
  input                 Combine datacard, or a model bundle (.json) written by 'build'
  --mass MASS           value substituted for $MASS in shapes lines
  --rmin RMIN           POI lower bound (default 0)
  --rmax RMAX           POI upper bound (default 20, as in Combine)
  --bin-integration {center,integral}
                        how parametric pdfs are evaluated on binned data: bin centre x width
                        (Combine/RooFit, default) or exact bin integrals
  --set-parameters SET_PARAMETERS
                        name=value,... (initial/fixed values)
  --freeze-parameters FREEZE_PARAMETERS
                        name,... parameters to fix at their values
  --freeze-nuisance-groups FREEZE_NUISANCE_GROUPS
                        group,... datacard groups to freeze
  --set-parameter-ranges SET_PARAMETER_RANGES
                        name=lo:hi,...

fit / output:
  --seed SEED           random seed (default 123456, as in Combine)
  --strategy {0,1,2}    Minuit strategy
  --tolerance TOLERANCE
                        Minuit tolerance (EDM goal 0.002*tol*0.5)
  --output OUTPUT, -o OUTPUT
                        result JSON (default pymodel_<command>.json)
  --plot                write plots for this command
  --plot-dir PLOT_DIR
  --verbose, -v
```

## `fc`

```
usage: pymodel roomodel fc [-h] [--mass MASS] [--rmin RMIN] [--rmax RMAX]
                           [--bin-integration {center,integral}] [--set-parameters SET_PARAMETERS]
                           [--freeze-parameters FREEZE_PARAMETERS]
                           [--freeze-nuisance-groups FREEZE_NUISANCE_GROUPS]
                           [--set-parameter-ranges SET_PARAMETER_RANGES] [--seed SEED]
                           [--strategy {0,1,2}] [--tolerance TOLERANCE] [--output OUTPUT] [--plot]
                           [--plot-dir PLOT_DIR] [--verbose] [--cl CL] [--grid GRID]
                           [--toys-per-point TOYS_PER_POINT] [--refine REFINE]
                           [--bypass-frequentist-fit] [--toys-file TOYS_FILE]
                           [--toy-index TOY_INDEX]
                           input

options:
  -h, --help            show this help message and exit
  --cl CL
  --grid GRID           r grid 'lo:hi:n' or list (default: from a likelihood scan)
  --toys-per-point TOYS_PER_POINT
  --refine REFINE       bisection points added around each edge
  --bypass-frequentist-fit
  --toys-file TOYS_FILE
  --toy-index TOY_INDEX

model:
  input                 Combine datacard, or a model bundle (.json) written by 'build'
  --mass MASS           value substituted for $MASS in shapes lines
  --rmin RMIN           POI lower bound (default 0)
  --rmax RMAX           POI upper bound (default 20, as in Combine)
  --bin-integration {center,integral}
                        how parametric pdfs are evaluated on binned data: bin centre x width
                        (Combine/RooFit, default) or exact bin integrals
  --set-parameters SET_PARAMETERS
                        name=value,... (initial/fixed values)
  --freeze-parameters FREEZE_PARAMETERS
                        name,... parameters to fix at their values
  --freeze-nuisance-groups FREEZE_NUISANCE_GROUPS
                        group,... datacard groups to freeze
  --set-parameter-ranges SET_PARAMETER_RANGES
                        name=lo:hi,...

fit / output:
  --seed SEED           random seed (default 123456, as in Combine)
  --strategy {0,1,2}    Minuit strategy
  --tolerance TOLERANCE
                        Minuit tolerance (EDM goal 0.002*tol*0.5)
  --output OUTPUT, -o OUTPUT
                        result JSON (default pymodel_<command>.json)
  --plot                write plots for this command
  --plot-dir PLOT_DIR
  --verbose, -v
```

## `significance`

```
usage: pymodel roomodel significance [-h] [--mass MASS] [--rmin RMIN] [--rmax RMAX]
                                     [--bin-integration {center,integral}]
                                     [--set-parameters SET_PARAMETERS]
                                     [--freeze-parameters FREEZE_PARAMETERS]
                                     [--freeze-nuisance-groups FREEZE_NUISANCE_GROUPS]
                                     [--set-parameter-ranges SET_PARAMETER_RANGES] [--seed SEED]
                                     [--strategy {0,1,2}] [--tolerance TOLERANCE]
                                     [--output OUTPUT] [--plot] [--plot-dir PLOT_DIR] [--verbose]
                                     [--method {asymptotic,toys}]
                                     [--toys-per-point TOYS_PER_POINT] [--bypass-frequentist-fit]
                                     [--toys-file TOYS_FILE] [--toy-index TOY_INDEX]
                                     input

options:
  -h, --help            show this help message and exit
  --method {asymptotic,toys}
  --toys-per-point TOYS_PER_POINT
  --bypass-frequentist-fit
  --toys-file TOYS_FILE
  --toy-index TOY_INDEX

model:
  input                 Combine datacard, or a model bundle (.json) written by 'build'
  --mass MASS           value substituted for $MASS in shapes lines
  --rmin RMIN           POI lower bound (default 0)
  --rmax RMAX           POI upper bound (default 20, as in Combine)
  --bin-integration {center,integral}
                        how parametric pdfs are evaluated on binned data: bin centre x width
                        (Combine/RooFit, default) or exact bin integrals
  --set-parameters SET_PARAMETERS
                        name=value,... (initial/fixed values)
  --freeze-parameters FREEZE_PARAMETERS
                        name,... parameters to fix at their values
  --freeze-nuisance-groups FREEZE_NUISANCE_GROUPS
                        group,... datacard groups to freeze
  --set-parameter-ranges SET_PARAMETER_RANGES
                        name=lo:hi,...

fit / output:
  --seed SEED           random seed (default 123456, as in Combine)
  --strategy {0,1,2}    Minuit strategy
  --tolerance TOLERANCE
                        Minuit tolerance (EDM goal 0.002*tol*0.5)
  --output OUTPUT, -o OUTPUT
                        result JSON (default pymodel_<command>.json)
  --plot                write plots for this command
  --plot-dir PLOT_DIR
  --verbose, -v
```

## `generate`

```
usage: pymodel roomodel generate [-h] [--mass MASS] [--rmin RMIN] [--rmax RMAX]
                                 [--bin-integration {center,integral}]
                                 [--set-parameters SET_PARAMETERS]
                                 [--freeze-parameters FREEZE_PARAMETERS]
                                 [--freeze-nuisance-groups FREEZE_NUISANCE_GROUPS]
                                 [--set-parameter-ranges SET_PARAMETER_RANGES] [--seed SEED]
                                 [--strategy {0,1,2}] [--tolerance TOLERANCE] [--output OUTPUT]
                                 [--plot] [--plot-dir PLOT_DIR] [--verbose] [--toys TOYS]
                                 [--expect-signal EXPECT_SIGNAL] [--toys-frequentist]
                                 [--bypass-frequentist-fit] [--toys-no-systematics]
                                 [--toys-file TOYS_FILE] [--toys-out TOYS_OUT]
                                 input

options:
  -h, --help            show this help message and exit
  --toys-out TOYS_OUT

model:
  input                 Combine datacard, or a model bundle (.json) written by 'build'
  --mass MASS           value substituted for $MASS in shapes lines
  --rmin RMIN           POI lower bound (default 0)
  --rmax RMAX           POI upper bound (default 20, as in Combine)
  --bin-integration {center,integral}
                        how parametric pdfs are evaluated on binned data: bin centre x width
                        (Combine/RooFit, default) or exact bin integrals
  --set-parameters SET_PARAMETERS
                        name=value,... (initial/fixed values)
  --freeze-parameters FREEZE_PARAMETERS
                        name,... parameters to fix at their values
  --freeze-nuisance-groups FREEZE_NUISANCE_GROUPS
                        group,... datacard groups to freeze
  --set-parameter-ranges SET_PARAMETER_RANGES
                        name=lo:hi,...

fit / output:
  --seed SEED           random seed (default 123456, as in Combine)
  --strategy {0,1,2}    Minuit strategy
  --tolerance TOLERANCE
                        Minuit tolerance (EDM goal 0.002*tol*0.5)
  --output OUTPUT, -o OUTPUT
                        result JSON (default pymodel_<command>.json)
  --plot                write plots for this command
  --plot-dir PLOT_DIR
  --verbose, -v

toys (Combine semantics):
  --toys TOYS, -t TOYS  number of toys (-1: Asimov dataset)
  --expect-signal EXPECT_SIGNAL
                        r used to generate toys (default 0)
  --toys-frequentist    fit nuisances to data and randomise global observables (Combine
                        --toysFrequentist)
  --bypass-frequentist-fit
                        frequentist toys around the pre-fit nuisance values
  --toys-no-systematics
                        do not randomise nuisances
  --toys-file TOYS_FILE
                        read datasets saved by 'generate' instead of generating
```

## `export`

```
usage: pymodel roomodel export [-h] [--mass MASS] [--rmin RMIN] [--rmax RMAX]
                               [--bin-integration {center,integral}]
                               [--set-parameters SET_PARAMETERS]
                               [--freeze-parameters FREEZE_PARAMETERS]
                               [--freeze-nuisance-groups FREEZE_NUISANCE_GROUPS]
                               [--set-parameter-ranges SET_PARAMETER_RANGES] [--seed SEED]
                               [--strategy {0,1,2}] [--tolerance TOLERANCE] [--output OUTPUT]
                               [--plot] [--plot-dir PLOT_DIR] [--verbose] --native-out NATIVE_OUT
                               input

options:
  -h, --help            show this help message and exit
  --native-out NATIVE_OUT

model:
  input                 Combine datacard, or a model bundle (.json) written by 'build'
  --mass MASS           value substituted for $MASS in shapes lines
  --rmin RMIN           POI lower bound (default 0)
  --rmax RMAX           POI upper bound (default 20, as in Combine)
  --bin-integration {center,integral}
                        how parametric pdfs are evaluated on binned data: bin centre x width
                        (Combine/RooFit, default) or exact bin integrals
  --set-parameters SET_PARAMETERS
                        name=value,... (initial/fixed values)
  --freeze-parameters FREEZE_PARAMETERS
                        name,... parameters to fix at their values
  --freeze-nuisance-groups FREEZE_NUISANCE_GROUPS
                        group,... datacard groups to freeze
  --set-parameter-ranges SET_PARAMETER_RANGES
                        name=lo:hi,...

fit / output:
  --seed SEED           random seed (default 123456, as in Combine)
  --strategy {0,1,2}    Minuit strategy
  --tolerance TOLERANCE
                        Minuit tolerance (EDM goal 0.002*tol*0.5)
  --output OUTPUT, -o OUTPUT
                        result JSON (default pymodel_<command>.json)
  --plot                write plots for this command
  --plot-dir PLOT_DIR
  --verbose, -v
```
