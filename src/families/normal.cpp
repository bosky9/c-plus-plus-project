#include "families/normal.hpp"

#include <random>
#include <chrono>
#include <cassert>

Normal::Normal(double mu, double sigma, std::string transform) : Family(transform), mu0{mu}, sigma0{sigma},
    param_no{2}, covariance_prior{false} {}

Eigen::MatrixXd* Normal::approximating_model(std::vector<double> beta, Eigen::MatrixXd T, Eigen::MatrixXd Z, Eigen::MatrixXd R, Eigen::MatrixXd Q,
                                             double h_approx, std::vector<double> data) {
    Eigen::MatrixXd H_mu[2];
    H_mu[0] = Eigen::MatrixXd::Constant(data.size(), data.size(), h_approx);
    H_mu[1] = Eigen::MatrixXd::Zero(data.size(), data.size());

    // Pointer to array (of size 2) of Eigen::MatrixXd
    return H_mu;
}

Eigen::MatrixXd* Normal::approximating_model_reg(std::vector<double> beta, Eigen::MatrixXd T, Eigen::MatrixXd Z, Eigen::MatrixXd R, Eigen::MatrixXd Q,
                                                 double h_approx, std::vector<double> data, std::vector<double> X, int state_no) {
    // It may be necessary to convert ulong to long
    Eigen::MatrixXd H_mu[2];
    H_mu[0] = Eigen::MatrixXd::Constant(data.size(), data.size(), h_approx);
    H_mu[1] = Eigen::MatrixXd::Zero(data.size(), data.size());

    // Pointer to array (of size 2) of Eigen::MatrixXd
    return H_mu;
}

// copy/move constructor may be needed
// what about the trasform of the Normal?
std::list<Normal::lv_to_build> Normal::build_latent_variables() {
    std::list<Normal::lv_to_build> lvs_to_build;
    lvs_to_build.push_back(Normal::lv_to_build{"Normal scale", new Normal{0, 3}, 0});
}

std::vector<double> Normal::draw_variable(double loc, double scale, double shape,
                                  double skewness, int nsims) {
    std::normal_distribution<double> my_normal{loc, scale};
    std::vector<double> sims(nsims);
    std::random_device rd{};
    std::mt19937 gen{rd()};
    for(int n = 0; n++; n < nsims)
        sims.at(n) = my_normal(gen);
    return sims;
}

std::vector<double> Normal::draw_variable_local(int size) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution(mu0, sigma0);
    std::vector<double> vars;
    if (size > 0) {
        for (int i = 0; i < size; i++) {
            vars.push_back(distribution(generator));
        }
    }
    return vars;
}

double Normal::logpdf(double mu) {
    if (!transform.empty()) {
        mu = _transform(mu);
    }
    return -log(sigma0) - (0.5 * std::pow(mu - mu0, 2)) / std::pow(sigma0, 2);
}

std::vector<double> Normal::markov_blanket(std::vector<double> y, std::vector<double> mean, double scale, double shape, double skewness) {
    std::vector<double> result;
    if (mean.size() == 1) {
        for (auto elem : y) {
            result.push_back(-log(scale) - (0.5 * std::pow(elem - mean.at(0), 2)) / std::pow(scale, 2));
        }
    } else {
        assert(y.size() == mean.size());
        for (size_t i = 0; i < y.size(); i++) {
            result.push_back(-log(scale) - (0.5 * std::pow(y.at(i) - mean.at(i), 2)) / std::pow(scale, 2));
        }
    }
    return result;
}

FamilyAttributes Normal::setup() {
    return {"Normal", [](double x){ return x; }, true, false, false, [](double x){ return x; }, true};
}

double Normal::neg_loglikelihood(std::vector<double> y, std::vector<double> mean, double scale, double shape, double skewness) {
    assert(!y.empty());
    double result = 0;
    std::vector<double> logpdf = Normal::markov_blanket(y, mean, scale, shape, skewness);
    for (auto elem : logpdf) {
        result += elem;
    }
    return result;
}

double Normal::pdf(double mu) {
    if (!transform.empty()) {
        mu = _transform(mu);
    }
    return (1.0/sigma0) * exp(-((0.5 * std::pow(mu - mu0, 2)) / std::pow(sigma0, 2)));
}

void Normal::vi_change_param(int index, double value) {
    if (index == 0)
        mu0 = value;
    else if (index == 1)
        sigma0 = exp(value);
}

double Normal::vi_return_param(int index) {
    if (index == 0)
        return mu0;
    else if (index == 1)
        return log(sigma0);
}

double Normal::vi_loc_score(double x) {
    return (x - mu0) / pow(sigma0, 2);
}

double Normal::vi_scale_score(double x) {
    return exp(-2 * log(sigma0)) * pow(x - mu0, 2) - 1;
}

double Normal::vi_score(double x, int index) {
    if (index == 0)
        return vi_loc_score(x);
    else if (index == 1)
        return vi_scale_score(x);
}