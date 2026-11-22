%% Independent MATLAB R2026a prototype for the equivariant score model
% This experiment is a research cross-check. It does not enter production.

scriptDir = fileparts(mfilename("fullpath"));
prototypeStarted = tic;
projectDir = fileparts(scriptDir);
dataPath = fullfile(projectDir, "data", "processed", "matches.csv");
outputDir = fullfile(projectDir, "artifacts", "matlab");
if ~isfolder(outputDir)
    mkdir(outputDir);
end

matches = readtable(dataPath, TextType="string", VariableNamingRule="preserve");
[evenFeatures, oddFeatures] = priorTeamFeatures(matches);
testYears = [1986, 1990, 1994, 1998, 2002, 2006, 2010];
alpha = 1.0;
rows = cell(numel(testYears), 5);

for foldIndex = 1:numel(testYears)
    testYear = testYears(foldIndex);
    trainMask = matches.Year < testYear;
    testMask = matches.Year == testYear;
    [beta, center, scale] = fitEquivariantPoisson( ...
        evenFeatures(trainMask, :), oddFeatures(trainMask, :), ...
        matches.("Home Team Goals")(trainMask), ...
        matches.("Away Team Goals")(trainMask), alpha);
    rates = predictEquivariantRates( ...
        evenFeatures(testMask, :), oddFeatures(testMask, :), beta, center, scale);
    probabilities = outcomeProbabilities(rates, 14);
    homeGoals = matches.("Home Team Goals")(testMask);
    awayGoals = matches.("Away Team Goals")(testMask);
    label = 1 + (homeGoals == awayGoals) + 2 * (homeGoals < awayGoals);
    selected = probabilities(sub2ind(size(probabilities), (1:sum(testMask))', label));
    observed = zeros(sum(testMask), 3);
    observed(sub2ind(size(observed), (1:sum(testMask))', label)) = 1;
    rows(foldIndex, :) = {testYear, -mean(log(max(selected, 1e-15))), ...
        mean(sum((probabilities - observed).^2, 2)), mean(label == maxIndex(probabilities)), ...
        sum(testMask)};
end

metrics = cell2table(rows, VariableNames=["test_year", "log_loss", ...
    "multiclass_brier", "accuracy", "n"]);
writetable(metrics, fullfile(outputDir, "advanced_prototype_metrics.csv"));

% Structural theorem checks use inputs unrelated to the fitted football data.
syntheticEven = [-1.0; 0.2; 1.4];
syntheticOdd = [0.7; -0.3; 1.1];
mu = 0.15;
evenCoefficient = 0.35;
oddCoefficient = -0.22;
homeRate = exp(mu + evenCoefficient * syntheticEven + oddCoefficient * syntheticOdd);
awayRate = exp(mu + evenCoefficient * syntheticEven - oddCoefficient * syntheticOdd);
swappedHome = exp(mu + evenCoefficient * syntheticEven - oddCoefficient * syntheticOdd);
swappedAway = exp(mu + evenCoefficient * syntheticEven + oddCoefficient * syntheticOdd);
swapError = max(abs([homeRate - swappedAway; awayRate - swappedHome]));
rateProductRange = range(homeRate .* awayRate);

baseGrid = independentGrid(1.65, 0.95, 14);
target = [0.41, 0.31, 0.28];
projected = minimumKLProjection(baseGrid, target);
projectedMargins = gridMargins(projected);
projectionError = max(abs(projectedMargins - target));
transposeTarget = [target(3), target(2), target(1)];
transposeProjection = minimumKLProjection(baseGrid', transposeTarget);
projectionSwapError = max(abs(transposeProjection - projected'), [], "all");

invariants = struct( ...
    "matlab_release", version("-release"), ...
    "swap_error", swapError, ...
    "rate_product_range", rateProductRange, ...
    "kl_margin_error", projectionError, ...
    "kl_swap_error", projectionSwapError, ...
    "all_passed", swapError < 1e-12 && rateProductRange > 1e-3 ...
        && projectionError < 1e-12 && projectionSwapError < 1e-12, ...
    "claim_scope", "independent prototype; not production evidence");
writeJson(fullfile(outputDir, "advanced_theory_invariants.json"), invariants);

card = struct( ...
    "experiment_id", "matlab_equivariant_prototype_v1", ...
    "data_version", "supplied World Cup CSVs through 2014", ...
    "time_split", "expanding-origin outer folds 1986-2010", ...
    "feature_set", "strictly prior tournament team scoring sums and differences", ...
    "preprocessing", "within-fold centering/scaling", ...
    "model_family", "penalized S2 even-odd Poisson GLM", ...
    "hyperparameters", struct("ridge_alpha", alpha), ...
    "random_seed", "not applicable; deterministic optimizer", ...
    "validation_metrics", ["log_loss", "multiclass_brier", "accuracy"], ...
    "runtime_seconds", toc(prototypeStarted), ...
    "selected_features", ["prior goals-for sum", "prior goals-conceded sum", ...
        "prior goals-for difference", "prior goals-conceded difference"], ...
    "artifact_paths", ["advanced_prototype_metrics.csv", ...
        "advanced_theory_invariants.json"], ...
    "production_dependency", false);
writeJson(fullfile(outputDir, "advanced_experiment_card.json"), card);

assert(invariants.all_passed, "Advanced MATLAB structural invariants failed.");
fprintf("Advanced MATLAB prototype complete: mean log loss %.6f, all invariants passed.\n", ...
    mean(metrics.log_loss));

function [even, odd] = priorTeamFeatures(matches)
    count = height(matches);
    even = zeros(count, 2);
    odd = zeros(count, 2);
    state = containers.Map("KeyType", "char", "ValueType", "any");
    years = unique(matches.Year, "sorted");
    for yearIndex = 1:numel(years)
        year = years(yearIndex);
        positions = find(matches.Year == year);
        for position = positions'
            home = char(matches.("Home Team Initials")(position));
            away = char(matches.("Away Team Initials")(position));
            homeState = getState(state, home);
            awayState = getState(state, away);
            homeFor = safeMean(homeState(2), homeState(1));
            awayFor = safeMean(awayState(2), awayState(1));
            homeAgainst = safeMean(homeState(3), homeState(1));
            awayAgainst = safeMean(awayState(3), awayState(1));
            even(position, :) = [homeFor + awayFor, homeAgainst + awayAgainst];
            odd(position, :) = [homeFor - awayFor, homeAgainst - awayAgainst];
        end
        for position = positions'
            home = char(matches.("Home Team Initials")(position));
            away = char(matches.("Away Team Initials")(position));
            homeGoals = matches.("Home Team Goals")(position);
            awayGoals = matches.("Away Team Goals")(position);
            state(home) = updateState(getState(state, home), homeGoals, awayGoals);
            state(away) = updateState(getState(state, away), awayGoals, homeGoals);
        end
    end
end

function value = getState(state, key)
    if isKey(state, key)
        value = state(key);
    else
        value = [0, 0, 0];
    end
end

function value = updateState(value, goalsFor, goalsAgainst)
    value = value + [1, goalsFor, goalsAgainst];
end

function value = safeMean(total, count)
    if count == 0
        value = 0;
    else
        value = total / count;
    end
end

function [beta, center, scale] = fitEquivariantPoisson(even, odd, homeGoals, awayGoals, alpha)
    numeric = [even, odd];
    center = mean(numeric, 1);
    scale = std(numeric, 0, 1);
    scale(scale < 1e-10) = 1;
    evenScaled = (even - center(1:size(even, 2))) ./ scale(1:size(even, 2));
    oddScaled = (odd - center(size(even, 2) + 1:end)) ./ ...
        scale(size(even, 2) + 1:end);
    design = [ones(size(even, 1), 1), evenScaled, oddScaled; ...
        ones(size(even, 1), 1), evenScaled, -oddScaled];
    response = [homeGoals; awayGoals];
    options = optimoptions("fminunc", Display="off", Algorithm="quasi-newton", ...
        MaxIterations=2000, OptimalityTolerance=1e-9);
    beta = fminunc(@(coefficient) poissonObjective(coefficient, design, response, alpha), ...
        zeros(size(design, 2), 1), options);
end

function [objective, gradient] = poissonObjective(beta, design, response, alpha)
    linear = design * beta;
    rate = exp(min(linear, 20));
    penalty = [0; beta(2:end)];
    objective = sum(rate - response .* linear) + 0.5 * alpha * sum(penalty .^ 2);
    gradient = design' * (rate - response) + alpha * penalty;
end

function rates = predictEquivariantRates(even, odd, beta, center, scale)
    evenWidth = size(even, 2);
    evenScaled = (even - center(1:evenWidth)) ./ scale(1:evenWidth);
    oddScaled = (odd - center(evenWidth + 1:end)) ./ scale(evenWidth + 1:end);
    homeDesign = [ones(size(even, 1), 1), evenScaled, oddScaled];
    awayDesign = [ones(size(even, 1), 1), evenScaled, -oddScaled];
    rates = exp([homeDesign * beta, awayDesign * beta]);
end

function probabilities = outcomeProbabilities(rates, maximumScore)
    count = size(rates, 1);
    probabilities = zeros(count, 3);
    for index = 1:count
        grid = independentGrid(rates(index, 1), rates(index, 2), maximumScore);
        probabilities(index, :) = gridMargins(grid);
    end
end

function grid = independentGrid(homeRate, awayRate, maximumScore)
    support = 0:maximumScore;
    grid = poisspdf(support, homeRate)' * poisspdf(support, awayRate);
    grid = grid / sum(grid, "all");
end

function margins = gridMargins(grid)
    margins = [sum(tril(grid, -1), "all"), trace(grid), sum(triu(grid, 1), "all")];
end

function projected = minimumKLProjection(grid, target)
    masks = {tril(true(size(grid)), -1), eye(size(grid), "logical"), ...
        triu(true(size(grid)), 1)};
    projected = zeros(size(grid));
    for outcome = 1:3
        source = sum(grid(masks{outcome}));
        if source <= 0 && target(outcome) > 0
            error("A requested outcome partition has zero source mass.");
        end
        if target(outcome) > 0
            projected(masks{outcome}) = grid(masks{outcome}) * target(outcome) / source;
        end
    end
end

function index = maxIndex(probabilities)
    [~, index] = max(probabilities, [], 2);
end

function writeJson(path, value)
    handle = fopen(path, "w");
    if handle < 0
        error("Cannot open JSON artifact for writing: %s", path);
    end
    cleaner = onCleanup(@() fclose(handle));
    fprintf(handle, "%s", jsonencode(value, PrettyPrint=true));
end
