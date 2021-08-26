#include "families/normal.hpp"

#include "multivariate_normal.hpp"

Normal::Normal(double mu, double sigma, const std::string& transform)
    : Family{transform}, _mu0{mu}, _sigma0{sigma}, _param_no{2}, _covariance_prior{false} {}

Normal::Normal(const Normal& normal) : Family(normal) {
    _mu0              = normal._mu0;
    _sigma0           = normal._sigma0;
    _param_no         = normal._param_no;
    _covariance_prior = normal._covariance_prior;
}

Normal::Normal(Normal&& normal) : Family(std::move(normal)) {
    _mu0                     = normal._mu0;
    _sigma0                  = normal._sigma0;
    _param_no                = normal._param_no;
    _covariance_prior        = normal._covariance_prior;
    normal._mu0              = 0;
    normal._sigma0           = 0;
    normal._param_no         = 0;
    normal._covariance_prior = false;
}

Normal& Normal::operator=(const Normal& normal) {
    if (this == &normal)
        return *this;
    Family::operator  =(normal);
    _mu0              = normal._mu0;
    _sigma0           = normal._sigma0;
    _param_no         = normal._param_no;
    _covariance_prior = normal._covariance_prior;
    return *this;
}

Normal& Normal::operator=(Normal&& normal) {
    _mu0                     = normal._mu0;
    _sigma0                  = normal._sigma0;
    _param_no                = normal._param_no;
    _covariance_prior        = normal._covariance_prior;
    normal._mu0              = 0;
    normal._sigma0           = 0;
    normal._param_no         = 0;
    normal._covariance_prior = false;
    Family::operator         =(std::move(normal));
    return *this;
}

/**
 * @brief Equal operator for Normal
 * @param family1 First object
 * @param family2 Second object
 * @return If the two objects are equal
 */
bool operator==(const Normal& normal1, const Normal& normal2) {
    return is_equal(normal1, normal2) && normal1._mu0 == normal2._mu0 && normal1._sigma0 == normal2._sigma0 &&
           normal1._param_no == normal2._param_no && normal1._covariance_prior == normal2._covariance_prior;
}

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

Eigen::VectorXd Normal::draw_variable_local(size_t size) const {
    return Mvn::random(_mu0, _sigma0, size);
}

double Normal::logpdf(double mu) {
    if (!_transform_name.empty())
        mu = _transform(mu);
    return -log(_sigma0) - (0.5 * std::pow(mu - _mu0, 2)) / std::pow(_sigma0, 2);
}

Eigen::VectorXd Normal::markov_blanket(const Eigen::VectorXd& y, const Eigen::VectorXd& means, double scale,
                                       double shape, double skewness) {
    return Mvn::logpdf(y, means, Eigen::Vector<double, 1>{scale});
}

FamilyAttributes Normal::setup() {
    return {"Normal", [](double x) { return x; }, true, false, false, [](double x) { return x; }, true};
}

double Normal::neg_loglikelihood(const Eigen::VectorXd& y, const Eigen::VectorXd& mean, double scale, double shape,
                                 double skewness) {
    assert(y.size() > 0);
    return Normal::markov_blanket(y, mean, scale, shape, skewness).sum();
}

double Normal::pdf(double mu) {
    if (!_transform_name.empty())
        mu = _transform(mu);
    return (1.0 / _sigma0) * exp(-((0.5 * std::pow(mu - _mu0, 2)) / std::pow(_sigma0, 2)));
}

void Normal::vi_change_param(int index, double value) {
    if (index == 0)
        _mu0 = value;
    else if (index == 1)
        _sigma0 = exp(value);
}

double Normal::vi_return_param(int index) const {
    if (index == 0)
        return _mu0;
    else if (index == 1)
        return log(_sigma0);
}

double Normal::vi_loc_score(double x) const {
    return (x - _mu0) / pow(_sigma0, 2);
}

double Normal::vi_scale_score(double x) const {
    return exp(-2 * log(_sigma0)) * pow(x - _mu0, 2) - 1;
}

double Normal::vi_score(double x, int index) const {
    if (index == 0)
        return vi_loc_score(x);
    else if (index == 1)
        return vi_scale_score(x);
}

short int Normal::get_param_no() const {
    return _param_no;
}