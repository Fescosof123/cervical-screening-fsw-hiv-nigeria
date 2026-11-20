# Cervical screening within HIV care for female sex workers in Nigeria

Analysis code for the manuscript "Cervical Screening Within HIV Care for Female Sex Workers in Nigeria: Coverage, Missing Results, and Yield in Program Data".

## Contents

- `analysis script.py`: the full analysis. It reads the programme line list and writes every number, table and figure used in the manuscript.

## How to run

```
python "analysis script.py" <line list, 8 November 2024 (.xlsx)> <output folder> [<line list, 10 April 2026 (.xlsx)>]
```

The third argument is optional and runs the follow-up check of missing results and treatment.

Requirements: Python 3.13, pandas 3.0, statsmodels 0.15, scipy 1.18, matplotlib 3.11, openpyxl.

Note on numbering: in the results workbook and log, "Table 2" is the regression table (Table 3 in the article) and "Table 3" is the table of results by year, method and state (Table 2 in the article).

## Data

The data are not included. They belong to the programme and contain sensitive information on a key population. De-identified data may be made available by the corresponding author on reasonable request and with the permission of the programme.

## Reporting

The study is reported according to STROBE (cross-sectional studies) and its RECORD extension.
