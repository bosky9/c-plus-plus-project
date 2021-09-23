#include "tsm.hpp"

TSM::TSM(std::string model_type) :
    _model_type{model_type},
    _latent_variables{model_type} {};

BBVIResults
TSM::_bbvi_fit(const std::function<double(Eigen::VectorXd)> &posterior, const std::string &optimizer, size_t iterations,
               bool map_start, size_t batch_size, std::optional<size_t> mini_batch, double learning_rate,
               bool record_elbo, bool quiet_progress) {
    Eigen::MatrixXd start_loc;
    Eigen::MatrixXd phi = _latent_variables.get_z_starting_values();
    // phi = kwargs.get('start',phi).copy()
    if (map_start && !mini_batch){
        //p = lbfgs search from d lib
        //start_loc = 0.8*p.x + 0.2*phi
    }
    else
        start_loc = phi;
    Eigen::MatrixXd start_ses{};

    for (uint i = 0; i < _latent_variables.get_z_list().size(); i++){
        Family* approx_dist = _latent_variables.get_z_list()[i].get_q();
        // isinstance(approx_dist, Normal): could use a str name
        if (start_ses.size() == 0){
            // need a way to set z_list[i]
        }
    }


}