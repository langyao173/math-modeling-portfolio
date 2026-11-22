function [service,electricity,share]=v3_annual_expectations(p,s,year)
% Learned beliefs from observed initial state and fixed in-training selection.
h=year-p.base_year;where=find(p.gdp_years==year,1);
if ~isempty(where)
    gdp=p.gdp_values(where);
else
    gdp=p.base_gdp;
    for t=p.base_year+1:year
        known=find(p.gdp_years==t,1);
        if ~isempty(known),gdp=p.gdp_values(known);else
            if t<=2030,j=1;elseif t<=2040,j=2;else,j=3;end
            gdp=gdp*(1+s.gdp_growth(j));
        end
    end
end
mult=1;if isfield(s,'demand_growth_multiplier'),mult=s.demand_growth_multiplier;end
eff=0;if isfield(s,'efficiency_improvement_per_year'),eff=s.efficiency_improvement_per_year;end
service=v3_annual_forecast_level(p.initial_service,gdp/p.base_gdp,h,p.expectations.service_fit,mult)*exp(-eff*h);
electricity=v3_annual_forecast_level(p.initial_electricity,gdp/p.base_gdp,h,p.expectations.electricity_fit,mult)*exp(-eff*h);
electricShare=min(.95,max(1e-8,electricity*.9/service));
fossil=log(p.initial_shares(1:3)')+p.share_logratio_drift(1:3)'*h;
fossil=exp(fossil-max(fossil));fossil=fossil/sum(fossil)*(1-electricShare);
share=[fossil;electricShare];
end
