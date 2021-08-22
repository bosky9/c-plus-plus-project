#include "inference/metropolis_sampler.hpp"

void metropolis_sampler(int sims_to_do, Eigen::MatrixXd& phi, const std::function<double(Eigen::VectorXd)>& posterior,
                        Eigen::VectorXd& a_rate, const Eigen::MatrixXd& rnums, const Eigen::VectorXd& crit) {

    double old_lik = -posterior(phi.row(0)); // float in Cython

    for (Eigen::Index i = 1; i < sims_to_do; i++) {
        Eigen::VectorXd phi_prop = phi.row(i - 1) + rnums.row(i);
        double post_prop         = -posterior(phi_prop);     // float in Cython
        double lik_rate          = exp(post_prop - old_lik); // float in Cython

        if (crit(i) < lik_rate) {
            phi.row(i) = phi_prop;
            a_rate(i)  = 1;
            old_lik    = post_prop;
        } else
            phi(i) = phi(i - 1);
    }
}