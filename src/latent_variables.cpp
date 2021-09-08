#include "latent_variables.hpp"

#include "families/family.hpp"
#include <string>
#include <vector>
#include <list>
#include <tuple>

LatentVariable::LatentVariable(const std::string& name, const Family& prior, const Family& q) :
      _name{name},
      _prior{prior},
      _transform{prior.get_transform()},
      _q{q}
{}

void LatentVariable::plot_z(double width, double height) {
    assert(_sample != nullptr || (_value != nullptr && _std != nullptr));
    if (_sample != nullptr) {
        //TODO
    }
    else {
        //TODO
    }
}

LatentVariables::LatentVariables(const std::string& model_name) :
      _model_name{model_name}
{
    _z_list = {};
    _z_indices = {};
}

friend std::ostream& operator<<(std::ostream& stream, const LatentVariables& latent_variables) {
    std::vector<std::string> z_names = get_z_names();
    std::vector<Family> priors = get_z_priors();
    std::pair<std::vector<std::string>,std::vector<std::string>> z_priors_names = get_z_priors_names();
    std::vector<std::string> prior_names = z_priors_names.first;
    std::vector<std::string> prior_z_names = z_priors_names.second;
    std::vector<std::string> vardist_names = get_z_approx_dist_names();
    std::vector<std::string> transforms = get_z_transforms_names();

    std::list<std::tuple<std::string, std::string, int>> fmt = {
            std::make_tuple("Index","z_index",8),
            std::make_tuple("Latent Variable","z_name",25),
            std::make_tuple("Prior","z_prior",15),
            std::make_tuple("Prior Hyperparameters","z_hyper",25),
            std::make_tuple("V.I. Dist","z_vardist",10),
            std::make_tuple("Transform","z_transform",10)
    };

    //TODO: Assegnare a stream il risultato di TablePrinter(fmt,ul="=")(z_row)

    stream << "";
    return stream;
}

void LatentVariables::add_z(const std::string& name, const Family& prior, const Family& q, bool index) {
    LatentVariable lv{name,prior,q};
    _z_list.push_back(std::move(lv));
    if (index)
        _z_indices[name] = { {"start",_z_list.size()-1}, {"end",_z_list.size()-1} };
}