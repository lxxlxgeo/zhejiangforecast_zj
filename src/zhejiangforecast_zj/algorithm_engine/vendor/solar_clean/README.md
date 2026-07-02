# solar_clean

Lightweight photovoltaic station data cleaning pipeline.

The first version avoids a hard pvlib dependency and implements conservative
rules that are useful for forecasting training samples:

- physical bounds for power and irradiance;
- night or very-low-irradiance high power checks;
- high-irradiance zero-power checks;
- power-vs-irradiance robust binned outlier checks;
- repeated/stuck daylight power checks;
- spike checks where power jumps while irradiance is nearly stable.

pvlib solar position, clear-sky and clearness-index features can be added later
for a stronger physical model when the runtime dependency is available.
