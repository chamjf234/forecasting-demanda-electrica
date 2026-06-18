# Servicio vivo de forecasting de demanda eléctrica

Pronóstico de demanda eléctrica horaria (región PJM, datos de la EIA) servido
como un sistema que se mantiene y monitorea solo. (Documentación completa y
demo: pendientes en planes de orquestación y dashboard.)

## Estado
En construcción. Plan actual: fundación de datos (ingesta + histórico versionado).

## Setup (desarrollo)

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1   |  Unix: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # instala el paquete `forecasting` (necesario para el CLI)
cp .env.example .env      # y pon tu EIA_API_KEY (https://www.eia.gov/opendata/register.php)
```

### Backfill inicial del histórico
```bash
python -m forecasting.bootstrap 2024-01-01T00 2024-06-01T00
```
