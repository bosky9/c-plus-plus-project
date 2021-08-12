#include "families/normal.hpp"


Normal::Normal(double mu, double sigma, const std::string& transform)
    : Family{transform}, mu0{mu}, sigma0{sigma}, param_no{2}, covariance_prior{false} {}

Eigen::MatrixXd* Normal::approximating_model(const std::vector<double>& beta, const Eigen::MatrixXd& T,
                                             const Eigen::MatrixXd& Z, const Eigen::MatrixXd& R,
                                             const Eigen::MatrixXd& Q, double h_approx,
                                             const std::vector<double>& data) {
    std::unique_ptr<Eigen::MatrixXd[]> H_mu(new Eigen::MatrixXd[2]);
    auto size = static_cast<Eigen::Index>(data.size());
    H_mu[0]   = Eigen::MatrixXd::Constant(size, size, h_approx);
    H_mu[1]   = Eigen::MatrixXd::Zero(size, size);

    // Pointer to array (of size 2) of Eigen::MatrixXd
    return H_mu.get();
}

Eigen::MatrixXd* Normal::approximating_model_reg(const std::vector<double>& beta, const Eigen::MatrixXd& T,
                                                 const Eigen::MatrixXd& Z, const Eigen::MatrixXd& R,
                                                 const Eigen::MatrixXd& Q, double h_approx,
                                                 const std::vector<double>& data, const std::vector<double>& X,
                                                 int state_no) {
    std::unique_ptr<Eigen::MatrixXd[]> H_mu(new Eigen::MatrixXd[2]);
    auto size = static_cast<Eigen::Index>(data.size());
    H_mu[0]   = Eigen::MatrixXd::Constant(size, size, h_approx);
    H_mu[1]   = Eigen::MatrixXd::Zero(size, size);

    // Pointer to array (of size 2) of Eigen::MatrixXd
    return H_mu.get();
}

// Copy/move constructor may be needed
// What about the transform of the Normal?
std::list<Normal::lv_to_build> Normal::build_latent_variables() {
    std::list<Normal::lv_to_build> lvs_to_build;
    lvs_to_build.push_back(Normal::lv_to_build{});
    return lvs_to_build;
}

std::vector<double> Normal::draw_variable(double loc, double scale, double shape, double skewness, int nsims) {
    std::normal_distribution<double> my_normal{loc, scale};
    std::vector<double> sims(nsims);
    std::random_device rd{};
    std::mt19937 gen{rd()};
    for (int n = 0; n++; n < nsims)
        sims.at(n) = my_normal(gen);
    return sims;
}

std::vector<double> Normal::draw_variable_local(int size) const {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution{mu0, sigma0};
    std::vector<double> vars;
    if (size > 0) {
        for (size_t i{0}; i < size; i++)
            vars.push_back(distribution(generator));
    }
    return vars;
}

double Normal::logpdf(double mu) {
    if (!transform.empty())
        mu = _transform(mu);
    return -log(sigma0) - (0.5 * std::pow(mu - mu0, 2)) / std::pow(sigma0, 2);
}

std::vector<double> Normal::markov_blanket(const std::vector<double>& y, const std::vector<double>& mean, double scale,
                                           double shape, double skewness) {
    std::vector<double> result;
    const double ONE_OVER_SQRT_2PI = 0.39894228040143267793994605993438;
    if (mean.size() == 1) {
        for (auto elem : y)
            result.push_back(log((ONE_OVER_SQRT_2PI/scale) * exp(-0.5*pow((elem-mean.at(0))/scale,2.0))));
    } else {
        assert(y.size() == mean.size());
        for (size_t i{0}; i < y.size(); i++)
            result.push_back(log((ONE_OVER_SQRT_2PI/scale) * exp(-0.5*pow((y.at(i)-mean.at(i))/scale,2.0))));
    }
    return result;
}

FamilyAttributes Normal::setup() {
    return {"Normal", [](double x) { return x; }, true, false, false, [](double x) { return x; }, true};
}

double Normal::neg_loglikelihood(const std::vector<double>& y, const std::vector<double>& mean, double scale,
                                 double shape, double skewness) {
    assert(!y.empty());
    double result{0};
    std::vector<double> logpdf{Normal::markov_blanket(y, mean, scale, shape, skewness)};
    for (auto elem : logpdf)
        result += elem;
    return result;
}

double Normal::pdf(double mu) {
    if (!transform.empty())
        mu = _transform(mu);
    return (1.0 / sigma0) * exp(-((0.5 * std::pow(mu - mu0, 2)) / std::pow(sigma0, 2)));
}

void Normal::vi_change_param(int index, double value) {
    if (index == 0)
        mu0 = value;
    else if (index == 1)
        sigma0 = exp(value);
}

double Normal::vi_return_param(int index) const {
    if (index == 0)
        return mu0;
    else if (index == 1)
        return log(sigma0);
}

double Normal::vi_loc_score(double x) const {
    return (x - mu0) / pow(sigma0, 2);
}

double Normal::vi_scale_score(double x) const {
    return exp(-2 * log(sigma0)) * pow(x - mu0, 2) - 1;
}

double Normal::vi_score(double x, int index) const {
    if (index == 0)
        return vi_loc_score(x);
    else if (index == 1)
        return vi_scale_score(x);
}