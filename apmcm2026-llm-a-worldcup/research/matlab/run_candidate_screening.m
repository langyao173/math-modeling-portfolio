function run_candidate_screening()
%RUN_CANDIDATE_SCREENING Reproducible MATLAB-only candidate cross-check.
% This experiment is optional. Final artifacts are regenerated in Python.

started = tic;
scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptDir);
rawPath = fullfile(projectRoot, 'data', 'raw', 'WorldCups_preview.csv');
matchPath = fullfile(projectRoot, 'data', 'raw', 'WorldCupMatches.csv');
playerPath = fullfile(projectRoot, 'data', 'raw', 'WorldCupPlayers.csv');
outputDir = fullfile(projectRoot, 'outputs', 'matlab');
if ~isfolder(outputDir)
    mkdir(outputDir);
end

opts = detectImportOptions(rawPath, 'TextType', 'string');
T = readtable(rawPath, opts);
attendance = str2double(erase(string(T.Attendance), '.')) / 1e6;
X = [double(T.QualifiedTeams), double(T.MatchesPlayed), attendance];
y = double(T.GoalsScored);
n = height(T);

modelNames = ["ols", "ridge", "poisson_glm"]';
predictions = nan(n, numel(modelNames));

for i = 1:n
    trainMask = true(n, 1);
    trainMask(i) = false;
    testX = X(i, :);

    ols = fitlm(X(trainMask, :), y(trainMask));
    predictions(i, 1) = predict(ols, testX);

    trainMean = mean(X(trainMask, :), 1);
    trainStd = std(X(trainMask, :), 0, 1);
    trainStd(trainStd == 0) = 1;
    standardizedTrain = (X(trainMask, :) - trainMean) ./ trainStd;
    standardizedTest = (testX - trainMean) ./ trainStd;
    augmentedTrain = [ones(sum(trainMask), 1), standardizedTrain];
    augmentedTest = [1, standardizedTest];
    lambda = 1.0;
    penalty = diag([0, ones(1, size(standardizedTrain, 2))]);
    beta = (augmentedTrain' * augmentedTrain + lambda * penalty) ...
        \ (augmentedTrain' * y(trainMask));
    predictions(i, 2) = augmentedTest * beta;

    poisson = fitglm(X(trainMask, :), y(trainMask), 'linear', ...
        'Distribution', 'poisson');
    predictions(i, 3) = predict(poisson, testX);
end

mae = mean(abs(predictions - y), 1)';
rmse = sqrt(mean((predictions - y).^2, 1))';
metrics = table(modelNames, mae, rmse, ...
    'VariableNames', {'model', 'loocv_mae', 'loocv_rmse'});
writetable(metrics, fullfile(outputDir, 'candidate_metrics.csv'));

finalOLS = fitlm(X, y, 'VarNames', ...
    {'QualifiedTeams', 'MatchesPlayed', 'AttendanceMillions', 'GoalsScored'});
coefficients = finalOLS.Coefficients;
coefficients.term = string(coefficients.Properties.RowNames);
coefficients = movevars(coefficients, 'term', 'Before', 1);
writetable(coefficients, fullfile(outputDir, 'ols_parity_coefficients.csv'), ...
    'WriteRowNames', false);

predictionTable = table(double(T.Year), y, predictions(:, 1), predictions(:, 2), ...
    predictions(:, 3), 'VariableNames', {'Year', 'observed', 'ols', 'ridge', 'poisson_glm'});
writetable(predictionTable, fullfile(outputDir, 'loocv_predictions.csv'));

% Independent parity checks for the Python production data contract.
matchOpts = detectImportOptions(matchPath, 'TextType', 'string');
matchOpts.VariableNamingRule = 'preserve';
Mraw = readtable(matchPath, matchOpts);
matchKeys = string(Mraw.RoundID) + "|" + string(Mraw.MatchID);
[~, uniqueMatchRows] = unique(matchKeys, 'stable');
M = Mraw(sort(uniqueMatchRows), :);
playerOpts = detectImportOptions(playerPath, 'TextType', 'string');
playerOpts.VariableNamingRule = 'preserve';
Praw = readtable(playerPath, playerOpts);
playerKeyColumns = strings(height(Praw), width(Praw));
for columnIndex = 1:width(Praw)
    columnName = Praw.Properties.VariableNames{columnIndex};
    if isstring(Praw.(columnName))
        Praw.(columnName) = strtrim(Praw.(columnName));
    end
    keyValues = strtrim(string(Praw.(columnName)));
    keyValues(ismissing(keyValues)) = "<MISSING>";
    playerKeyColumns(:, columnIndex) = keyValues;
end
playerKeys = join(playerKeyColumns, char(31), 2);
[~, uniquePlayerRows] = unique(playerKeys, 'stable');
P = Praw(sort(uniquePlayerRows), :);
fullTimeTotal = double(M.("Home Team Goals")) + double(M.("Away Team Goals"));
halfTimeTotal = double(M.("Half-time Home Goals")) + ...
    double(M.("Half-time Away Goals"));
halfFullCorrelation = corr(halfTimeTotal, fullTimeTotal, 'Rows', 'complete');

stageText = lower(strtrim(string(M.Stage)));
canonicalStage = strings(height(M), 1);
canonicalStage(contains(stageText, 'group')) = "group_stage";
canonicalStage(contains(stageText, 'first round') | ...
    contains(stageText, 'preliminary round') | ...
    contains(stageText, 'round of 16')) = "round_of_16";
canonicalStage(contains(stageText, 'quarter')) = "quarter_final";
canonicalStage(contains(stageText, 'semi')) = "semi_final";
canonicalStage(contains(stageText, 'third')) = "third_place";
canonicalStage(stageText == "final") = "final";
if any(canonicalStage == "")
    error('Unmapped stage labels remain in MATLAB parity check.');
end
stageOrder = ["group_stage", "round_of_16", "quarter_final", ...
    "semi_final", "final", "third_place"]';
stageMatches = zeros(numel(stageOrder), 1);
stageGoalSum = zeros(numel(stageOrder), 1);
stageMeanGoals = zeros(numel(stageOrder), 1);
for i = 1:numel(stageOrder)
    mask = canonicalStage == stageOrder(i);
    stageMatches(i) = sum(mask);
    stageGoalSum(i) = sum(fullTimeTotal(mask));
    stageMeanGoals(i) = mean(fullTimeTotal(mask));
end
stageParity = table(stageOrder, stageMatches, stageGoalSum, stageMeanGoals, ...
    'VariableNames', {'canonical_stage', 'matches', 'goal_sum', 'mean_goals'});
writetable(stageParity, fullfile(outputDir, 'parity_stage_statistics.csv'));

macroValues = [double(T.GoalsScored), ...
    double(T.GoalsScored) ./ double(T.MatchesPlayed), ...
    double(T.QualifiedTeams), double(T.MatchesPlayed)];
indicator = ["total_goals", "goals_per_match", ...
    "qualified_teams", "matches_played"]';
meanValue = mean(macroValues, 1)';
medianValue = median(macroValues, 1)';
minimum = min(macroValues, [], 1)';
maximum = max(macroValues, [], 1)';
sampleStd = std(macroValues, 0, 1)';
observations = repmat(height(T), 4, 1);
descriptiveParity = table(indicator, meanValue, medianValue, minimum, maximum, ...
    sampleStd, observations, 'VariableNames', {'indicator', 'mean', 'median', ...
    'minimum', 'maximum', 'sample_std', 'n'});
writetable(descriptiveParity, ...
    fullfile(outputDir, 'parity_descriptive_statistics.csv'));

year2014 = double(M.Year) == 2014;
attendance2014 = double(M.Attendance(year2014));
parity = struct();
parity.raw_match_rows = height(Mraw);
parity.canonical_match_rows = height(M);
parity.raw_player_rows = height(Praw);
parity.canonical_player_rows = height(P);
parity.matches_2014 = sum(year2014);
parity.goals_2014 = sum(fullTimeTotal(year2014));
parity.attendance_2014 = sum(attendance2014, 'omitnan');
parity.halftime_fulltime_correlation = halfFullCorrelation;
parity.attendance_parse_examples = [590549, 1045246, 3386810];
parity.stage_rows = table2struct(stageParity);
parity.tolerance = 1e-10;
writelines(jsonencode(parity, 'PrettyPrint', true), ...
    fullfile(outputDir, 'parity_summary.json'));

card = struct();
card.experiment_id = 'matlab_aggregate_candidate_screening_v1';
card.matlab_version = version;
card.data_file = 'data/raw/WorldCups_preview.csv';
card.data_rows = n;
card.data_cutoff = 2014;
card.time_split = 'leave-one-tournament-out diagnostic';
card.feature_set = {'QualifiedTeams', 'MatchesPlayed', 'AttendanceMillions'};
card.preprocessing = 'fold-local standardization for ridge only';
card.model_family = cellstr(modelNames);
card.hyperparameters = struct('ridge_lambda', 1.0);
card.random_seed = 'not_applicable_deterministic';
card.validation_metrics = table2struct(metrics);
card.runtime_seconds = toc(started);
card.selected_features = card.feature_set;
card.artifact_paths = {
    'outputs/matlab/candidate_metrics.csv', ...
    'outputs/matlab/ols_parity_coefficients.csv', ...
    'outputs/matlab/loocv_predictions.csv', ...
    'outputs/matlab/parity_descriptive_statistics.csv', ...
    'outputs/matlab/parity_stage_statistics.csv', ...
    'outputs/matlab/parity_summary.json'};
card.production_dependency = false;
card.evidence_boundary = ...
    'Candidate cross-check only; promoted methods must be reimplemented and rerun in Python.';
writelines(jsonencode(card, 'PrettyPrint', true), ...
    fullfile(outputDir, 'experiment_card.json'));

fprintf('MATLAB candidate screening complete: %d tournaments, %.3f seconds.\n', ...
    n, card.runtime_seconds);
end
