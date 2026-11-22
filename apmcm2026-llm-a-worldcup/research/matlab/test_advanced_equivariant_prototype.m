function tests = test_advanced_equivariant_prototype
% Independent regression tests for the advanced MATLAB experiment card.
tests = functiontests(localfunctions);
end

function testPrototypeArtifactsAndInvariants(testCase)
scriptDir = fileparts(mfilename("fullpath"));
prototypePath = fullfile(scriptDir, "advanced_equivariant_prototype.m");
evalin("base", "run(""" + prototypePath + """)");
projectDir = fileparts(scriptDir);
outputDir = fullfile(projectDir, "artifacts", "matlab");
metricsPath = fullfile(outputDir, "advanced_prototype_metrics.csv");
invariantPath = fullfile(outputDir, "advanced_theory_invariants.json");
cardPath = fullfile(outputDir, "advanced_experiment_card.json");
verifyTrue(testCase, isfile(metricsPath));
verifyTrue(testCase, isfile(invariantPath));
verifyTrue(testCase, isfile(cardPath));
metrics = readtable(metricsPath);
verifyEqual(testCase, height(metrics), 7);
verifyTrue(testCase, all(isfinite(metrics.log_loss)));
verifyTrue(testCase, all(metrics.log_loss > 0));
invariants = jsondecode(fileread(invariantPath));
verifyTrue(testCase, invariants.all_passed);
verifyLessThan(testCase, invariants.swap_error, 1e-12);
verifyGreaterThan(testCase, invariants.rate_product_range, 1e-3);
verifyLessThan(testCase, invariants.kl_margin_error, 1e-12);
verifyLessThan(testCase, invariants.kl_swap_error, 1e-12);
card = jsondecode(fileread(cardPath));
verifyFalse(testCase, card.production_dependency);
verifyEqual(testCase, card.time_split, 'expanding-origin outer folds 1986-2010');
end
