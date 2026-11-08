# Data availability

The monthly operational dataset for ATO1 Marche Nord (2017–2021) was provided by the service operators **Marche Multiservizi S.p.A.** and **ASET S.p.A.** It is subject to their data-sharing conditions and is **not included in this repository**. The dataset comprises:

- spring abstraction (`Sorgenti`);
- well-field abstraction (`Pozzi`);
- served population (`Popolazione`);
- treated drinking-water / potabilization volume (`Potabilizzatori`);
- the derived civil-demand (`Fabbisogni`) and network-loss (`Perdite`) series.

The processed series can be made available by the corresponding author on reasonable request and with the permission of the data providers.

To re-run the notebooks, place the file here as `data/Dataset.xlsx`. It has one sheet with the columns `Data, Sorgenti, Pozzi, Popolazione, Potabilizzatori, Fabbisogni, Perdite` (60 monthly rows, January 2017 – December 2021; volumes in m³/month). Network losses are the balance residual:

```
Perdite = Sorgenti + Pozzi + Potabilizzatori - Fabbisogni
```

In the source workbook, `Perdite` is an Excel formula (`=E2+C2+B2-F2`, i.e. Potabilizzatori + Pozzi + Sorgenti − Fabbisogni). The notebooks read the **stored values** of that column (`pandas.read_excel`). If the workbook has been written by a tool that does not store formula results (e.g. `openpyxl`), the column reads as empty. In that case, open and re-save the file in Excel or LibreOffice before running the notebooks. `figures/make_figures.py` recomputes `Perdite` from the identity and is not affected.

## Other withheld files

These files embed the observed values and are therefore also withheld (they are listed in `.gitignore`):

| File | Content |
|---|---|
| `output_configc/configc_predictions_long.csv` | Per-origin observed/predicted values, parsimonious pipeline |
| `output_part2_5seed/configa_extended_predictions_long.csv` | Per-origin observed/predicted values, extended pipeline |
| `output_part2_5seed/integrated_dataset.csv` | Operational series merged with covariates |

The aggregated and per-seed metric tables in `output_*/` contain no observed values and are distributed.

## Public data

The cached covariates in `data_cache/` are derived from public sources:

- **ERA5-Land reanalysis** (Muñoz-Sabater et al., 2021). *Contains modified Copernicus Climate Change Service information (1991–2021).*
- **ISTAT tourism statistics** (*Movimento dei clienti negli esercizi ricettivi*), © Istituto Nazionale di Statistica.

These files remain subject to their providers' terms.
