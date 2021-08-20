#include "inference/metropolis_hastings.hpp"

MetropolisHastings::MetropolisHastings(std::function<double(Eigen::VectorXd)>& posterior, double scale, int nsims,
                                       const Eigen::VectorXd& initials,
                                       const std::optional<Eigen::MatrixXd>& cov_matrix, int thinning,
                                       bool warm_up_period, // TODO: TSM model_object = nullptr,
                                       bool quiet_progress)
    : _posterior{posterior}, _scale{scale}, _nsims{(1 + warm_up_period) * nsims * thinning}, _initials{initials},
      _param_no{initials.size()}, _thinning{thinning}, _warm_up_period{warm_up_period}, _quiet_progress{
                                                                                                quiet_progress} {

    _phi        = Eigen::MatrixXd::Zero(_nsims, _param_no);
    _phi.row(0) = _initials;
    _cov_matrix = cov_matrix.value_or(Eigen::MatrixXd::Identity(_param_no, _param_no) * _initials.cwiseAbs());

    // if (model_object == nullptr)
    //  _model = model_object;
}

double MetropolisHastings::tune_scale(double acceptance, double scale) {
    if (acceptance > 0.8)
        scale *= 2.0;
    else if (acceptance <= 0.8 && acceptance > 0.4)
        scale *= 1.3;
    else if (acceptance < 0.234 && acceptance > 0.1)
        scale *= (1 / 1.3);
    else if (acceptance <= 0.1 && acceptance > 0.05)
        scale *= 0.4;
    else if (acceptance <= 0.05 && acceptance > 0.01)
        scale *= 0.2;
    else if (acceptance <= 0.01)
        scale *= 0.1;
    return scale;
}
