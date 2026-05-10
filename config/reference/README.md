# Optional reference data for dashboards

Place CSV files in this directory when you need features that are not in the IRENA extract.

## `gdp_nominal_usd.csv`

Used to express IRENA `public_flows_2022_usd_m` (million USD) as a share of nominal GDP.

| Column | Type | Description |
|--------|------|-------------|
| `year` | int | Calendar year |
| `iso3` | str | ISO 3166-1 alpha-3 (e.g. `GBR`, `DEU`) |
| `gdp_nominal_usd` | float | GDP in current USD for that year (same units as your source, typically IMF WEO or World Bank) |

One row per `(year, iso3)`. Copy from [IMF WEO](https://www.imf.org/en/Publications/SPWEOS) or World Bank national accounts.

If this file is missing, the dashboard still loads; GDP-ratio charts show a notice.

## `population.csv`

Used to weight OECD or peer-country aggregates for SDG 7.b.1 (W per inhabitant) comparisons.

| Column | Type | Description |
|--------|------|-------------|
| `year` | int | Year |
| `iso3` | str | ISO3 country code |
| `population` | float | Mid-year population |

Source: UN World Population Prospects or World Bank `SP.POP.TOTL`.

If missing, SDG peer comparisons fall back to an unweighted median across selected high-income countries.

## `cpi_uk.csv`

Used by the **UK Energy Transition Integrated Brief** page to deflate nominal £ series (solar PV £/kW, RHI £ payments) to a common 2024 base before fitting learning rates and unit-economics.

| Column | Type | Description |
|--------|------|-------------|
| `year` | int | Calendar year |
| `cpi_index_2024_base` | float | UK CPI rebased so that 2024 ≡ 100 |

Source: ONS series `D7BT` (CPI INDEX 00 : ALL ITEMS 2015=100) — rebase by dividing each year's value by the 2024 value and multiplying by 100. A starter file (`cpi_uk.csv.example`) ships in this folder; copy it to `cpi_uk.csv` and replace the placeholder row.

If `cpi_uk.csv` is missing, the page proceeds with **nominal £** and a banner notes the caveat.
