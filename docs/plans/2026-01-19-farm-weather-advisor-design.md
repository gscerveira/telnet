# Farm Weather Advisor - Design Document

**Date:** 2026-01-19
**Status:** Approved
**Repo:** `farm-weather-advisor` (new, separate from telnet)

## Overview

"Conselheiro Agrícola" - A Streamlit web app for Brazilian farmers to get crop planting recommendations based on 3-month precipitation forecasts from Open-Meteo's Seasonal Forecast API.

**Target audience:** Investors/stakeholders (demo prototype)
**Language:** Brazilian Portuguese (all user-facing text)
**Timeline:** This week (3 days)

## Core User Flow

1. **Select location** - User clicks on a map of Brazil or enters city name
2. **View forecast** - App fetches 3-month precipitation forecast from Open-Meteo
3. **See recommendation** - Based on forecast (wet/dry/normal), app suggests suitable crops

## Data Flow

```
User selects location
       ↓
Open-Meteo Seasonal API (ECMWF SEAS5)
       ↓
3-month precipitation totals + anomaly
       ↓
Classify: "Wetter than normal" / "Drier than normal" / "Normal"
       ↓
Match to crop recommendations (hardcoded lookup table)
       ↓
Display results with confidence level
```

## Technical Architecture

### Project Structure

```
farm-weather-advisor/
├── app.py                 # Main Streamlit app
├── api/
│   └── openmeteo.py       # Open-Meteo API client
├── data/
│   ├── crops.json         # Crop recommendations by precipitation category
│   └── brazil_cities.json # City name → lat/lon lookup
├── components/
│   ├── map.py             # Brazil map selector (Plotly)
│   └── forecast.py        # Forecast chart component
├── i18n/
│   └── pt_br.py           # All Portuguese strings
├── requirements.txt
└── README.md
```

### Dependencies

- `streamlit` - UI framework
- `requests` - API calls
- `plotly` - Charts and Brazil map
- `pandas` - Data handling

### Open-Meteo API

```
GET https://seasonal-api.open-meteo.com/v1/seasonal
  ?latitude=-5.0
  &longitude=-45.0
  &daily=precipitation_sum
  &forecast_months=3
```

Returns 51 ensemble members - show median + 10th/90th percentile range.

### Deployment

Streamlit Community Cloud (free) - connect GitHub repo for automatic deployment.

## Crop Recommendation Logic

### Precipitation Categories

Compare forecasted precipitation against historical climatology:

| Category | Condition | Portuguese Label |
|----------|-----------|------------------|
| Dry | < -20% anomaly | "Previsão de seca" |
| Normal | -20% to +20% | "Chuvas dentro da média" |
| Wet | > +20% anomaly | "Previsão de chuvas acima da média" |

### Crop Lookup Table

```json
{
  "dry": {
    "recommended": ["sorgo", "milheto", "feijão-caupi"],
    "avoid": ["arroz", "mandioca"]
  },
  "normal": {
    "recommended": ["milho", "soja", "feijão"],
    "avoid": []
  },
  "wet": {
    "recommended": ["arroz", "mandioca", "hortaliças"],
    "avoid": ["feijão", "amendoim"]
  }
}
```

Each crop includes: name, icon, one-line reason.

### Confidence Display

Using 51 ensemble members:
- "Alta confiança" (>70% agreement)
- "Confiança moderada" (50-70%)
- "Incerto" (<50%)

## UI Design

### Landing Page

```
┌─────────────────────────────────────────────────────┐
│  🌱 Conselheiro Agrícola                            │
│  Planeje seu plantio com previsões sazonais         │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌─────────────────────────────────────┐            │
│  │         [Mapa do Brasil]            │            │
│  │        Clique para selecionar       │            │
│  └─────────────────────────────────────┘            │
│                                                     │
│  Ou digite o nome da cidade:                        │
│  ┌─────────────────────────┐  [Consultar]          │
│  └─────────────────────────┘                        │
└─────────────────────────────────────────────────────┘
```

### Results Page

```
┌─────────────────────────────────────────────────────┐
│  ← Voltar    📍 Imperatriz, MA                      │
├─────────────────────────────────────────────────────┤
│  PREVISÃO PARA OS PRÓXIMOS 3 MESES                  │
│  [Gráfico de barras: precipitação mensal]           │
│                                                     │
│  🔴 Previsão de seca (Alta confiança)               │
│  Precipitação 30% abaixo da média esperada          │
├─────────────────────────────────────────────────────┤
│  ✅ CULTURAS RECOMENDADAS                           │
│  ┌────────┐ ┌────────┐ ┌────────┐                  │
│  │ Sorgo  │ │Milheto │ │Caupi   │                  │
│  └────────┘ └────────┘ └────────┘                  │
│                                                     │
│  ⚠️ EVITAR NESTE PERÍODO                           │
│  Arroz, Mandioca                                   │
└─────────────────────────────────────────────────────┘
```

### Styling

- Clean white background
- Green accent color (#2E7D32)
- Modern sans-serif font
- Streamlit theming + minimal custom CSS

## Scope

### In Scope (MVP)

- Brazil map location picker + city search
- Open-Meteo API integration (seasonal forecast)
- Precipitation forecast chart (3 months)
- Dry/normal/wet classification with confidence
- Crop recommendations (~10 crops)
- All text in Brazilian Portuguese
- Deploy to Streamlit Cloud

### Out of Scope

- User accounts / saved locations
- Historical data comparison
- Multiple forecast variables (temperature)
- Actual agronomic advice
- Mobile app

## Timeline

| Day | Milestone |
|-----|-----------|
| 1 | Repo setup, API integration, basic UI |
| 2 | Map component, forecast chart, styling |
| 3 | Polish, deploy, README |

## Data Sources

- [Open-Meteo Seasonal Forecast API](https://open-meteo.com/en/docs/seasonal-forecast-api) - Free, ECMWF SEAS5 data, 51 ensemble members, 7 months ahead
