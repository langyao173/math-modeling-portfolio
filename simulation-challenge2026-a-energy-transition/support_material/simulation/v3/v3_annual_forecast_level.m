function value=v3_annual_forecast_level(initial,gdpRatio,elapsed,fit,growthMultiplier)
if nargin<5,growthMultiplier=1;end
if strcmp(fit.mode,'increment3')
    value=max(initial+fit.increment*elapsed*growthMultiplier,1e-9);
else
    value=initial*gdpRatio^(fit.beta*growthMultiplier)*exp(fit.drift*elapsed);
end
end
