"""Read-only audit of the fixed-penalty interval Gaussian M step.

Small independent synthetic moment calculations only. No beer generator is fit,
no parent implementation is changed, and no candidate-transfer model is scored.
"""
import os
for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
EXP = HERE.parent / "experiments/generation_moments"
sys.path.insert(0, str(EXP))
from penalized_interval_em import exact_m_step, expected_penalized_loss, expected_scatter


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    rng = np.random.default_rng(20261024091)
    records = []
    for p, n, retained, kappa, nu in [(3, 2, 8, 2., 20.), (5, 7, 32, 2., 20.),
                                    (17, 2, 4, 2., 20.), (6, 28, 8, .3, 5.)]:
        a = rng.normal(size=(p, p))
        prior_cov = a@a.T/p + .4*np.eye(p)
        prior_mean = rng.normal(size=p)
        latent = rng.normal(size=(retained, n, p)) + rng.normal(size=(1, 1, p))
        total = latent.mean(axis=0).sum(axis=0)
        second = np.einsum("rip,riq->pq", latent, latent)/retained
        empirical_mean = latent.mean(axis=(0, 1))
        centered = latent-empirical_mean
        within_direct = np.einsum("rip,riq->pq", centered, centered)/retained
        mu, cov = exact_m_step(total, second, n, prior_mean, prior_cov, kappa, nu)
        full_scatter_direct = np.einsum("rip,riq->pq", latent-mu, latent-mu)/retained
        full_scatter_direct += nu*prior_cov + kappa*np.outer(mu-prior_mean, mu-prior_mean)
        correction = n*kappa/(n+kappa)*np.outer(empirical_mean-prior_mean, empirical_mean-prior_mean)
        scatter_decomposed = within_direct+nu*prior_cov+correction
        objective = expected_penalized_loss(total, second, n, mu, cov, prior_mean, prior_cov, kappa, nu)
        independent_objective = .5*((n+nu)*np.linalg.slogdet(cov)[1]
            + np.trace(np.linalg.solve(cov, full_scatter_direct)))
        complete_square_errors, gaps = [], []
        for _ in range(80):
            candidate_mu = mu+rng.normal(size=p)
            b = rng.normal(size=(p, p))/np.sqrt(p)
            candidate_cov = cov+b@b.T+.1*np.eye(p)
            direct = np.einsum("rip,riq->pq", latent-candidate_mu, latent-candidate_mu)/retained
            direct += nu*prior_cov+kappa*np.outer(candidate_mu-prior_mean, candidate_mu-prior_mean)
            predicted = scatter_decomposed+(n+kappa)*np.outer(candidate_mu-mu, candidate_mu-mu)
            complete_square_errors.append(np.max(abs(direct-predicted)))
            gaps.append(expected_penalized_loss(total, second, n, candidate_mu, candidate_cov,
                prior_mean, prior_cov, kappa, nu)-objective)
        numerical_loss_gap = numerical_parameter_error = None
        if p <= 5:
            tri = np.tril_indices(p)
            diagonal = tri[0] == tri[1]
            def unpack(theta):
                mean = theta[:p]
                lower = np.zeros((p, p))
                lower[tri] = theta[p:]
                lower[np.diag_indices(p)] = np.exp(theta[p:][diagonal])
                return mean, lower, lower@lower.T
            def criterion_gradient(theta):
                mean, lower, covariance = unpack(theta)
                inverse = np.linalg.inv(covariance)
                scatter = np.einsum("rip,riq->pq", latent-mean, latent-mean)/retained
                scatter += nu*prior_cov+kappa*np.outer(mean-prior_mean, mean-prior_mean)
                value = .5*((n+nu)*np.linalg.slogdet(covariance)[1]+np.trace(inverse@scatter))
                mean_grad = inverse@((n+kappa)*mean-total-kappa*prior_mean)
                covariance_grad = .5*((n+nu)*inverse-inverse@scatter@inverse)
                lower_grad = (2*covariance_grad@lower)[tri]
                lower_grad[diagonal] *= np.diag(lower)
                return float(value), np.r_[mean_grad, lower_grad]
            initial_lower = np.linalg.cholesky(prior_cov)
            theta0 = np.r_[prior_mean, initial_lower[tri]]
            theta0[p:][diagonal] = np.log(np.diag(initial_lower))
            result = minimize(criterion_gradient, theta0, jac=True, method="BFGS",
                options={"gtol":1e-9, "maxiter":2000})
            optimized_mean, _, optimized_cov = unpack(result.x)
            numerical_loss_gap = float(result.fun-objective)
            numerical_parameter_error = float(max(np.max(abs(optimized_mean-mu)), np.max(abs(optimized_cov-cov))))
            assert abs(numerical_loss_gap) < 1e-8
            assert numerical_parameter_error < 1e-6
        inv = np.linalg.inv(cov)
        mean_gradient = inv@((n+kappa)*mu-total-kappa*prior_mean)
        covariance_gradient = .5*((n+nu)*inv-inv@full_scatter_direct@inv)
        rec = dict(p=p, n=n, retained=retained, kappa=kappa, nu=nu,
            within_scatter_minimum_eigenvalue=float(np.linalg.eigvalsh(within_direct)[0]),
            posterior_covariance_minimum_eigenvalue=float(np.linalg.eigvalsh(cov)[0]),
            scatter_direct_vs_decomposition_max_error=float(np.max(abs(full_scatter_direct-scatter_decomposed))),
            scatter_parent_vs_direct_max_error=float(np.max(abs(expected_scatter(total, second, n, mu, prior_mean, prior_cov, kappa, nu)-full_scatter_direct))),
            complete_square_max_error=float(max(complete_square_errors)),
            objective_independent_error=float(abs(objective-independent_objective)),
            mean_gradient_max=float(np.max(abs(mean_gradient))),
            covariance_gradient_max=float(np.max(abs(covariance_gradient))),
            normal_equation_covariance_max_error=float(np.max(abs((n+nu)*cov-full_scatter_direct))),
            random_alternative_minimum_loss_increase=float(min(gaps)),
            numerical_optimizer_loss_gap=numerical_loss_gap,
            numerical_optimizer_parameter_max_error=numerical_parameter_error)
        assert rec["posterior_covariance_minimum_eigenvalue"] > 0
        assert rec["scatter_direct_vs_decomposition_max_error"] < 1e-10
        assert rec["complete_square_max_error"] < 1e-10
        assert rec["mean_gradient_max"] < 1e-10
        assert rec["covariance_gradient_max"] < 1e-9
        assert min(gaps) >= -1e-9
        records.append(rec)
    result = dict(date="2026-10-24", audit_scope="Independent algebra and small synthetic numerical checks; no beer generator fits",
        code_sha256={name:sha(EXP/name) for name in ["penalized_interval_em.py", "penalized_em_derivation.md", "run_penalized_em.py"]},
        checks=records, joint_m_step_correct=True, full_second_moments_required=True,
        coefficient="n*kappa/(n+kappa)", finite_e_does_not_prove_observed_likelihood_monotonicity=True,
        exact_e_guarantee_is_for_penalized_observed_objective=True,
        final_saved_statistics_allow_after_loss_recalculation=True,
        final_saved_statistics_include_old_parameters=
            all(key in (EXP/"penalized_interval_em.py").read_text() for key in ["old_mean=old_mean[k].copy()", "old_cov=old_cov[k].copy()"]),
        final_saved_statistics_alone_require_old_parameters_to_recalculate_before_loss=True)
    (HERE/"penalized_interval_em_independent_review.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
